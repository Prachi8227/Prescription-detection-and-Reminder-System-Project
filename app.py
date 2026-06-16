from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory, jsonify, make_response
from collections import OrderedDict
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import os
import json
import time
import datetime
import csv
import io
import math
import logging
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import traceback
from zoneinfo import ZoneInfo

import requests
import re
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler  # type: ignore
from werkzeug.utils import secure_filename
from fpdf import FPDF

from ocr_module.image_trainer import ImageTrainer
from ocr_module.prescription_ocr import (
    process_prescription,
    evaluate_accuracy,
    SUPPORTED_LANGUAGES,
    DEFAULT_OCR_LANG_CODES,
    RECOMMENDED_MULTILINGUAL_CODES,
    normalize_language_codes,
    get_language_labels,
)
# OCR thresholds
HANDWRITING_QUALITY_THRESHOLD = 0.70
HANDWRITING_SELECTION_MARGIN = 0.10
HANDWRITING_MIN_SCORE = 0.50
MEDICAL_SIGNAL_THRESHOLD = 0.60

# from ocr_module.enhanced_ocr import (
#     HANDWRITING_QUALITY_THRESHOLD = 80,
#     HANDWRITING_SELECTION_MARGIN,
#     HANDWRITING_MIN_SCORE,
#     MEDICAL_SIGNAL_THRESHOLD,
# )

# Load environment variables from .env (if present)
load_dotenv()


app = Flask(__name__)
app.secret_key = 'your_secret_key'

logging.basicConfig(level=logging.INFO)

INDIA_TZ = ZoneInfo("Asia/Kolkata")

UI_LANGUAGES = OrderedDict([
    ("en", {"label": "English", "google_code": "en", "ocr_codes": ["eng"]}),
    ("hi", {"label": "Hindi", "google_code": "hi", "ocr_codes": ["eng", "hin"]}),
    ("mr", {"label": "Marathi", "google_code": "mr", "ocr_codes": ["eng", "mar"]}),
    ("kn", {"label": "Kannada", "google_code": "kn", "ocr_codes": ["eng", "kan"]}),
])
DEFAULT_UI_LANGUAGE = "en"

EMAIL_HOST = os.environ.get('EMAIL_HOST')
EMAIL_PORT = int(os.environ.get('EMAIL_PORT', 587))
# These must be provided via environment variables, never hard-coded
EMAIL_USERNAME = os.environ.get('EMAIL_USERNAME')      # e.g. your SMTP/Brevo login
EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD')      # e.g. app password or SMTP key
EMAIL_FROM = os.environ.get('EMAIL_FROM', EMAIL_USERNAME if EMAIL_USERNAME else None)
EMAIL_USE_TLS = os.environ.get('EMAIL_USE_TLS', 'true').lower() != 'false'

# Brevo (Sendinblue) transactional email support
BREVO_API_KEY = os.environ.get('BREVO_API_KEY')
BREVO_SENDER_NAME = os.environ.get('BREVO_SENDER_NAME', 'MediScribe AI')

EMAIL_SMTP_ENABLED = all([
    EMAIL_HOST, 
    EMAIL_PORT, 
    EMAIL_USERNAME, 
    EMAIL_PASSWORD, 
    EMAIL_FROM])
BREVO_ENABLED = bool(BREVO_API_KEY and EMAIL_FROM)
EMAIL_ENABLED = EMAIL_SMTP_ENABLED or BREVO_ENABLED

scheduler = None

# Custom Jinja2 filter for timestamp conversion
@app.template_filter('timestamp_to_date')
def timestamp_to_date(timestamp):
    """Convert a Unix timestamp to a formatted date string"""
    dt = datetime.datetime.fromtimestamp(timestamp)
    return dt.strftime('%B %d, %Y at %I:%M %p')

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff', 'webp'}

# Ensure upload and results directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Helper function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def build_file_url(file_path):
    """Return a publicly accessible URL for a stored file."""
    if not file_path:
        return None

    filename = os.path.basename(file_path)

    try:
        target_path = os.path.abspath(file_path)
        uploads_root = os.path.abspath(app.config['UPLOAD_FOLDER'])
        results_root = os.path.abspath(app.config['RESULTS_FOLDER'])

        if os.path.commonpath([target_path, uploads_root]) == uploads_root:
            return url_for('uploaded_file', filename=filename)

        if os.path.commonpath([target_path, results_root]) == results_root:
            return url_for('result_file', filename=filename)
    except (ValueError, OSError):
        # ValueError can be raised if paths are on different drives; ignore and fall back to existence checks
        pass

    uploads_candidate = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(uploads_candidate):
        return url_for('uploaded_file', filename=filename)

    results_candidate = os.path.join(app.config['RESULTS_FOLDER'], filename)
    if os.path.exists(results_candidate):
        return url_for('result_file', filename=filename)

    return None


@app.context_processor
def inject_ui_language_metadata():
    preferred_language = session.get('preferred_language', DEFAULT_UI_LANGUAGE)
    if preferred_language not in UI_LANGUAGES:
        preferred_language = DEFAULT_UI_LANGUAGE
    return {
        'ui_languages': UI_LANGUAGES,
        'current_language_code': preferred_language,
        'current_language_label': UI_LANGUAGES[preferred_language]['label'],
        'default_ui_language': DEFAULT_UI_LANGUAGE,
        'ui_language_codes': list(UI_LANGUAGES.keys()),
    }

DB_NAME = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'database.db')
OCR_SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ocr_module', 'ocr_config.json')
DEFAULT_OCR_SETTINGS = {
    'mode': 'auto',
    'quality_threshold': HANDWRITING_QUALITY_THRESHOLD,
    'medical_signal_threshold': MEDICAL_SIGNAL_THRESHOLD,
    'selection_margin': HANDWRITING_SELECTION_MARGIN,
    'min_score': HANDWRITING_MIN_SCORE,
}


def load_ocr_settings():
    settings = DEFAULT_OCR_SETTINGS.copy()
    try:
        if os.path.exists(OCR_SETTINGS_FILE):
            with open(OCR_SETTINGS_FILE, 'r') as f:
                data = json.load(f)
            if isinstance(data, dict):
                for key in settings:
                    if key in data:
                        settings[key] = data[key]
    except Exception as exc:
        logging.error("Failed to load OCR settings: %s", exc)
    return settings


def save_ocr_settings(new_settings):
    os.makedirs(os.path.dirname(OCR_SETTINGS_FILE), exist_ok=True)
    with open(OCR_SETTINGS_FILE, 'w') as f:
        json.dump(new_settings, f, indent=2)

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                     password TEXT NOT NULL,
                is_admin INTEGER DEFAULT 0,
                address TEXT,
                mobile_no TEXT,
                blood_group TEXT,
                pin_code TEXT,
                email_id TEXT,
                gender TEXT
            )
        ''')

    # Check if the admin user already exists
    c.execute("SELECT id FROM users WHERE username = ?", ('admin',))
    admin_exists = c.fetchone()

    if not admin_exists:
        # Insert default admin user if not exists
        c.execute("INSERT INTO users (username, password, is_admin, address, mobile_no, blood_group, pin_code, email_id, gender) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  ('admin', generate_password_hash('admin123'), 1, 'Default Address', '0000000000', 'O+', '000000', 'admin@example.com', 'Other'))
    else:
        # Ensure the admin user's is_admin status is always 1 if it already exists
        c.execute("UPDATE users SET is_admin = 1 WHERE username = ?", ('admin',))
        
    # Create reminders table if not exists
    c.execute('''
        CREATE TABLE IF NOT EXISTS reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            medication TEXT NOT NULL,
            dosage TEXT,
            frequency TEXT NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT,
            time_of_day TEXT,
            last_notification_sent TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    c.execute("PRAGMA table_info(reminders)")
    reminder_columns = [column[1] for column in c.fetchall()]
    if 'last_notification_sent' not in reminder_columns:
        c.execute("ALTER TABLE reminders ADD COLUMN last_notification_sent TEXT")
        
    conn.commit()
    conn.close()

init_db()


def parse_date(date_str):
    if not date_str:
        return None
    for fmt in ('%Y-%m-%d', '%d-%m-%Y', '%Y/%m/%d'):
        try:
            return datetime.datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None


def parse_time_string(time_str):
    if not time_str:
        return None
    normalized = time_str.strip()
    for fmt in ('%H:%M', '%H:%M:%S', '%I:%M %p', '%I:%M%p'):
        try:
            return datetime.datetime.strptime(normalized, fmt).time()
        except ValueError:
            continue
    return None


def format_display_date(date_str):
    parsed = parse_date(date_str)
    if not parsed:
        return None
    return parsed.strftime('%b %d, %Y')


def extract_durations(text):
    if not text:
        return []

    patterns = [
        r"\bfor\s+(\d+\s+(?:day|days|week|weeks|month|months))\b",
        r"\b(\d+\s+(?:day|days|week|weeks|month|months))\b",
        r"\buntil\s+(finished|complete|gone)\b",
    ]

    results = []
    lowered = text.lower()
    for pattern in patterns:
        matches = re.findall(pattern, lowered)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0]
            cleaned = match.strip()
            if cleaned and cleaned not in results:
                results.append(cleaned)
    return results


def build_structured_prescriptions(results_dict):
    raw_text = results_dict.get('cleaned_text') or results_dict.get('raw_text') or ''

    # Parse numbered prescription rows from trained/OCR text (e.g. 1) PAN HD *)
    try:
        from ocr_module.enhanced_ocr import extract_table_prescription_medicines
        table_meds = extract_table_prescription_medicines(raw_text)
        if table_meds:
            return [
                {
                    'name': m['name'],
                    'dosage': m.get('dosage') or 'As directed',
                    'frequency': m.get('frequency') or 'As prescribed',
                    'duration': m.get('duration') or 'Until finished',
                    'instruction': m.get('instruction') or 'Follow doctor instructions',
                }
                for m in table_meds
            ]
    except Exception:
        pass

    # Use saved structured rows from trained OCR or prescription-table parsing
    existing = results_dict.get('structured_prescriptions') or []
    use_saved = (
        existing
        and (
            results_dict.get('extraction_mode') in ('prescription_table', 'trained')
            or results_dict.get('is_trained')
            or results_dict.get('trained_match')
        )
    )
    if use_saved:
        structured = []
        seen = set()
        for med in existing:
            if isinstance(med, dict):
                name = (med.get('name') or '').strip()
            else:
                name = str(med).strip()
            key = name.lower()
            if not name or key in seen:
                continue
            seen.add(key)
            if isinstance(med, dict):
                structured.append({
                    'name': name,
                    'dosage': med.get('dosage') or 'As directed',
                    'frequency': med.get('frequency') or 'As prescribed',
                    'duration': med.get('duration') or 'Until finished',
                    'instruction': med.get('instruction') or 'Follow doctor instructions',
                })
            else:
                structured.append({
                    'name': name,
                    'dosage': 'As directed',
                    'frequency': 'As prescribed',
                    'duration': 'Until finished',
                    'instruction': 'Follow doctor instructions',
                })
        if structured:
            return structured

    # Handle case where results might come from trained database with different structure
    medications = results_dict.get('medications') or []
    dosages = results_dict.get('dosages') or []
    frequencies = results_dict.get('frequencies') or []
    routes = results_dict.get('routes') or []
    durations = results_dict.get('durations') or extract_durations(raw_text)
    
    # Handle case where medications might be in medicines_strict format (new enhanced OCR format)
    if not medications and 'medicines_strict' in results_dict and results_dict['medicines_strict']:
        medicines_strict = results_dict['medicines_strict'].get('medicines', [])
        medications = [med.get('name', 'Unknown') for med in medicines_strict]
        dosages = [med.get('strength', 'As directed') for med in medicines_strict]
        frequencies = [med.get('frequency', 'As prescribed') for med in medicines_strict]
        routes = [med.get('route', 'Follow doctor instructions') for med in medicines_strict]
        durations = [med.get('duration', 'Until finished') for med in medicines_strict]
    
    # If no medications found, try to extract basic medication info from text using enhanced patterns
    if not medications and raw_text:
        # Enhanced pattern to find potential medications
        # Look for common medication-related patterns
        medication_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:tablet|capsule|syrup|injection|tab|cap|syp|inj)\b',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)(?:\s+\d+(?:mg|ml|g)?)\b',
            r'\b(?:tab|tablet|cap|capsule|syp|syrup)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
            r'\b([A-Z][a-z]+\d+)\b',  # Medicine names with numbers (e.g., "Paracetamol500")
            r'\b([A-Z]{2,}[0-9]{2,})\b',  # Common medicine formats (e.g., "CIP500", "DOLO650")
            r'\b([A-Z]{2,}-[A-Z0-9]+)\b',  # Medicine names with hyphens (e.g., "MED-X")
            r'\b([A-Z][a-z]{3,}(?:\s+[A-Z][a-z]{2,})*)\s+\d+\s*(?:mg|ml|g)\b',  # Medicine name followed by dosage
            r'\b(Tab|Cap|Syp|Inj)\s+([A-Z][a-z]+)\b',  # Indian prescription format (e.g., "Tab Paracip")
            r'\b(Tab|Cap|Syp|Inj)\s+([A-Z][a-z]{4,})\b',  # Indian prescription format with longer names
            r'\b([A-Z][a-z]{3,})\s+(\d+mg|\d+ml|\d+g)\b',  # Medicine names with dosage after (e.g., "Aten 50mg")
            r'\b([A-Z][a-z]+\d{3,})\b',  # Medicine names with strength numbers (e.g., "Dolo650")
            r'\b([A-Z][a-z]{4,})\s+(\d{1,3}(?:\.\d{1,2})?\s*(?:mg|ml|g))\b',  # Common Indian medicine formats (e.g., "Rabeprazole 20mg")
            r'\b([A-Z]{5,}[0-9]*)\b',  # Standalone capitalized medicine names (e.g., "ATENOLOL", "CIPROFLOXACIN")
        ]
        
        for pattern in medication_patterns:
            matches = re.findall(pattern, raw_text, re.IGNORECASE)
            for match in matches:
                med_name = match if isinstance(match, str) else match[0] if len(match) > 0 else match
                # If it's a tuple with two groups (like "Tab Paracip"), take the medicine name part
                if isinstance(match, tuple) and len(match) > 1:
                    med_name = match[1]  # Take the second group which is the medicine name
                elif isinstance(match, tuple) and len(match) > 0:
                    med_name = match[0]
                
                # Basic cleaning
                med_name = re.sub(r'\s+', ' ', med_name.strip())
                if len(med_name) > 2 and med_name.lower() not in ['the', 'and', 'for', 'with', 'take', 'have', 'not', 'this', 'that', 'will', 'should', 'would', 'could', 'may', 'might', 'must', 'can', 'does', 'did', 'do', 'is', 'are', 'was', 'were', 'been', 'being', 'be', 'has', 'had', 'having', 'get', 'got', 'getting', 'make', 'made', 'making', 'go', 'went', 'going', 'come', 'came', 'coming', 'see', 'saw', 'seeing', 'know', 'knew', 'knowing', 'think', 'thought', 'thinking', 'say', 'said', 'saying', 'tell', 'told', 'telling', 'ask', 'asked', 'asking', 'give', 'gave', 'giving', 'put', 'turn', 'turned', 'turning', 'keep', 'kept', 'keeping', 'let', 'lets', 'letting', 'begin', 'began', 'beginning', 'start', 'started', 'starting', 'continue', 'continued', 'continuing', 'try', 'tried', 'trying', 'need', 'needed', 'needing', 'want', 'wanted', 'wanting', 'like', 'liked', 'liking', 'seem', 'seemed', 'seeming', 'become', 'became', 'becoming', 'leave', 'left', 'leaving', 'feel', 'felt', 'feeling', 'appear', 'appeared', 'appearing', 'look', 'looked', 'looking', 'hear', 'heard', 'hearing', 'play', 'played', 'playing', 'run', 'ran', 'running', 'move', 'moved', 'moving', 'live', 'lived', 'living', 'believe', 'believed', 'believing', 'hold', 'held', 'holding', 'bring', 'brought', 'bringing', 'happen', 'happened', 'happening', 'write', 'wrote', 'writing', 'sit', 'sat', 'sitting', 'stand', 'stood', 'standing', 'lose', 'lost', 'losing', 'pay', 'paid', 'paying', 'meet', 'met', 'meeting', 'include', 'included', 'including', 'set', 'setting', 'learn', 'learned', 'learning', 'change', 'changed', 'changing', 'lead', 'led', 'leading', 'understand', 'understood', 'understanding', 'watch', 'watched', 'watching', 'follow', 'followed', 'following', 'stop', 'stopped', 'stopping', 'create', 'created', 'creating', 'speak', 'spoke', 'speaking', 'read', 'spend', 'spent', 'spending', 'grow', 'grew', 'growing', 'open', 'opened', 'opening', 'walk', 'walked', 'walking', 'win', 'won', 'winning', 'teach', 'taught', 'teaching', 'offer', 'offered', 'offering', 'remember', 'remembered', 'remembering', 'consider', 'considered', 'considering', 'buy', 'bought', 'buying', 'serve', 'served', 'serving', 'die', 'died', 'dying', 'send', 'sent', 'sending', 'build', 'built', 'building', 'stay', 'stayed', 'staying', 'fall', 'fell', 'falling', 'cut', 'rise', 'rose', 'rising', 'drive', 'drove', 'driving', 'break', 'broke', 'breaking', 'choose', 'chose', 'choosing', 'forget', 'forgot', 'forgetting', 'drink', 'drank', 'drinking', 'eat', 'ate', 'eating', 'find', 'found', 'finding', 'fly', 'flew', 'flying', 'hide', 'hid', 'hiding', 'hit', 'lay', 'laid', 'laying', 'lie', 'ring', 'rang', 'ringing', 'shake', 'shook', 'shaking', 'sing', 'sang', 'singing', 'sink', 'sank', 'sinking', 'stick', 'stuck', 'sticking', 'strike', 'struck', 'striking', 'tear', 'tore', 'tearing', 'throw', 'threw', 'throwing', 'wake', 'woke', 'waking', 'wear', 'wore', 'wearing', 'diagnosis', 'symptom', 'symptoms', 'treatment', 'therapy', 'procedure', 'operation', 'surgery', 'test', 'exam', 'checkup', 'consultation', 'appointment', 'followup', 'review', 'monitoring', 'monitor', 'check', 'screening', 'screen', 'evaluation', 'assessment', 'prescribe', 'prescribing', 'prescriber', 'pharmacist', 'pharmacy', 'pharmaceutical', 'pharmaceuticals', 'pharma', 'medicinal', 'medicinals', 'therapeutic', 'therapeutics', 'clinical', 'clinically', 'medical', 'medically', 'health', 'healthcare', 'care', 'hospital', 'clinic', 'doctor', 'physician', 'specialist', 'nurse', 'nursing', 'patient', 'patients', 'client', 'clients', 'person', 'people', 'human', 'humans', 'individual', 'individuals', 'subject', 'subjects']:
                    medications.append(med_name.title())
        
        # Additional pattern matching for common medicine formats in Indian prescriptions
        # Look for patterns like "Tab. Paracip", "Cap. Amlokind", etc.
        indian_patterns = [
            r'\b(Tab|Cap|Syp|Inj)\.?\s+([A-Z][a-z]{3,})\b',
            r'\b([A-Z][a-z]{4,})\s+(?:\d+(?:mg|ml|g)?)\b',
            r'\b([A-Z]{5,}[0-9]*)\b',  # Standalone capitalized medicine names
        ]
        
        for pattern in indian_patterns:
            matches = re.findall(pattern, raw_text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple) and len(match) > 1:
                    med_name = match[1]  # Take the medicine name part
                elif isinstance(match, tuple) and len(match) > 0:
                    med_name = match[0]
                else:
                    med_name = match
                
                med_name = re.sub(r'\s+', ' ', med_name.strip())
                # Extended skip words list to prevent false positives
                skip_words = ['the', 'and', 'for', 'with', 'take', 'have', 'not', 'this', 'that', 'will', 'should', 'would', 'could', 'may', 'might', 'must', 'can', 'does', 'did', 'do', 'is', 'are', 'was', 'were', 'been', 'being', 'be', 'has', 'had', 'having', 'get', 'got', 'getting', 'make', 'made', 'making', 'go', 'went', 'going', 'come', 'came', 'coming', 'see', 'saw', 'seeing', 'know', 'knew', 'knowing', 'think', 'thought', 'thinking', 'say', 'said', 'saying', 'tell', 'told', 'telling', 'ask', 'asked', 'asking', 'give', 'gave', 'giving', 'put', 'turn', 'turned', 'turning', 'keep', 'kept', 'keeping', 'let', 'lets', 'letting', 'begin', 'began', 'beginning', 'start', 'started', 'starting', 'continue', 'continued', 'continuing', 'try', 'tried', 'trying', 'need', 'needed', 'needing', 'want', 'wanted', 'wanting', 'like', 'liked', 'liking', 'seem', 'seemed', 'seeming', 'become', 'became', 'becoming', 'leave', 'left', 'leaving', 'feel', 'felt', 'feeling', 'appear', 'appeared', 'appearing', 'look', 'looked', 'looking', 'hear', 'heard', 'hearing', 'play', 'played', 'playing', 'run', 'ran', 'running', 'move', 'moved', 'moving', 'live', 'lived', 'living', 'believe', 'believed', 'believing', 'hold', 'held', 'holding', 'bring', 'brought', 'bringing', 'happen', 'happened', 'happening', 'write', 'wrote', 'writing', 'sit', 'sat', 'sitting', 'stand', 'stood', 'standing', 'lose', 'lost', 'losing', 'pay', 'paid', 'paying', 'meet', 'met', 'meeting', 'include', 'included', 'including', 'set', 'setting', 'learn', 'learned', 'learning', 'change', 'changed', 'changing', 'lead', 'led', 'leading', 'understand', 'understood', 'understanding', 'watch', 'watched', 'watching', 'follow', 'followed', 'following', 'stop', 'stopped', 'stopping', 'create', 'created', 'creating', 'speak', 'spoke', 'speaking', 'read', 'spend', 'spent', 'spending', 'grow', 'grew', 'growing', 'open', 'opened', 'opening', 'walk', 'walked', 'walking', 'win', 'won', 'winning', 'teach', 'taught', 'teaching', 'offer', 'offered', 'offering', 'remember', 'remembered', 'remembering', 'consider', 'considered', 'considering', 'buy', 'bought', 'buying', 'serve', 'served', 'serving', 'die', 'died', 'dying', 'send', 'sent', 'sending', 'build', 'built', 'building', 'stay', 'stayed', 'staying', 'fall', 'fell', 'falling', 'cut', 'rise', 'rose', 'rising', 'drive', 'drove', 'driving', 'break', 'broke', 'breaking', 'choose', 'chose', 'choosing', 'forget', 'forgot', 'forgetting', 'drink', 'drank', 'drinking', 'eat', 'ate', 'eating', 'find', 'found', 'finding', 'fly', 'flew', 'flying', 'hide', 'hid', 'hiding', 'hit', 'lay', 'laid', 'laying', 'lie', 'ring', 'rang', 'ringing', 'shake', 'shook', 'shaking', 'sing', 'sang', 'singing', 'sink', 'sank', 'sinking', 'stick', 'stuck', 'sticking', 'strike', 'struck', 'striking', 'tear', 'tore', 'tearing', 'throw', 'threw', 'throwing', 'wake', 'woke', 'waking', 'wear', 'wore', 'wearing', 'diagnosis', 'symptom', 'symptoms', 'treatment', 'therapy', 'procedure', 'operation', 'surgery', 'test', 'exam', 'checkup', 'consultation', 'appointment', 'followup', 'review', 'monitoring', 'monitor', 'check', 'screening', 'screen', 'evaluation', 'assessment', 'prescribe', 'prescribing', 'prescriber', 'pharmacist', 'pharmacy', 'pharmaceutical', 'pharmaceuticals', 'pharma', 'medicinal', 'medicinals', 'therapeutic', 'therapeutics', 'clinical', 'clinically', 'medical', 'medically', 'health', 'healthcare', 'care', 'hospital', 'clinic', 'doctor', 'physician', 'specialist', 'nurse', 'nursing', 'patient', 'patients', 'client', 'clients', 'person', 'people', 'human', 'humans', 'individual', 'individuals', 'subject', 'subjects']
                if len(med_name) > 2 and med_name.lower() not in skip_words:
                    medications.append(med_name.title())
        
        # If still no medications, add a default entry if text suggests medications
        if not medications and re.search(r'\b(?:medication|prescription|tablet|capsule|syrup|take|dose|mg|ml|g)\b', raw_text, re.IGNORECASE):
            medications = ['Medication Not Clearly Identified']
            dosages = ['See prescription text']
            frequencies = ['As prescribed']
    
        # If we still have no structured medications but there is OCR text,
    # fall back to showing meaningful lines from the prescription as
    # "medications" so the doctor/patient always sees some result.
    if not medications and raw_text:
        lines = []
        for line in raw_text.splitlines():
            line = line.strip()
            # Skip very short / empty lines
            if len(line) < 3:
                continue
            # Require at least one letter to avoid pure numbers / noise
            if not re.search(r'[A-Za-z]', line):
                continue
            lines.append(line)

        # De‑duplicate lines case‑insensitively
        seen = set()
        fallback_meds = []
        for line in lines:
            key = line.lower()
            if key in seen:
                continue
            seen.add(key)
            fallback_meds.append(line)

        if fallback_meds:
            medications = fallback_meds
            dosages = ['See prescription text'] * len(fallback_meds)
            frequencies = ['As prescribed'] * len(fallback_meds)
            durations = ['Until finished'] * len(fallback_meds)
            routes = ['Follow doctor instructions'] * len(fallback_meds)

    # Ensure we have at least some data to display
    if not medications:
        medications = ['No medications detected']
        dosages = ['N/A']
        frequencies = ['N/A']
        durations = ['N/A']
        routes = ['N/A']

    # Remove duplicate medicine names before pairing fields.
    seen_med_names = set()
    unique_medications = []
    for med in medications:
        med_key = re.sub(r'\s+', ' ', str(med).strip()).lower()
        if not med_key or med_key in seen_med_names:
            continue
        seen_med_names.add(med_key)
        unique_medications.append(med)
    medications = unique_medications
    
    structured = []
    max_len = max(len(medications), len(dosages), len(frequencies), len(durations), len(routes), 1)
    
    for idx in range(max_len):
        name = medications[idx] if idx < len(medications) else f"Medication {idx + 1}"
        dosage = dosages[idx] if idx < len(dosages) else (dosages[0] if dosages else 'As directed')
        frequency = frequencies[idx] if idx < len(frequencies) else (frequencies[0] if frequencies else 'As prescribed')
        duration = durations[idx] if idx < len(durations) else (durations[0] if durations else 'Until finished')
        instruction = routes[idx] if idx < len(routes) else (routes[0] if routes else 'Follow doctor instructions')
        
        # Skip entries with empty or placeholder values, but allow "Medication Not Clearly Identified"
        if name and (name != 'Not mentioned' and name != 'Medication 1') or name in ['Medication Not Clearly Identified', 'No medications detected']:
            structured.append({
                'name': (
                    name
                    if results_dict.get('extraction_mode') == 'prescription_table'
                    or name in ['Medication Not Clearly Identified', 'No medications detected']
                    else (name.title() if isinstance(name, str) else name)
                ),
                'dosage': dosage,
                'frequency': frequency,
                'duration': duration,
                'instruction': instruction,
            })
    
    # Keep only one row per medicine name to avoid repeated entries
    # caused by overlapping OCR extraction patterns.
    deduped_structured = []
    by_name = {}

    def detail_score(item):
        defaults = {
            'as directed',
            'as prescribed',
            'until finished',
            'follow doctor instructions',
            'n/a',
            'not mentioned',
            '—',
            '',
        }
        score = 0
        for field in ('dosage', 'frequency', 'duration', 'instruction'):
            value = str(item.get(field, '')).strip().lower()
            if value and value not in defaults:
                score += 1
        return score

    for entry in structured:
        name_key = re.sub(r'\s+', ' ', str(entry.get('name', '')).strip()).lower()
        if not name_key:
            continue
        existing = by_name.get(name_key)
        if existing is None:
            by_name[name_key] = entry
            deduped_structured.append(entry)
            continue
        if detail_score(entry) > detail_score(existing):
            replace_index = deduped_structured.index(existing)
            deduped_structured[replace_index] = entry
            by_name[name_key] = entry

    return deduped_structured


def send_email(subject, body, recipient):
    if not EMAIL_ENABLED:
        logging.debug("Email disabled - skipping send to %s", recipient)
        return False

    try:
        # Prefer Brevo HTTP API if configured
        if BREVO_ENABLED:
            payload = {
                "sender": {
                    "email": EMAIL_FROM,
                    "name": BREVO_SENDER_NAME,
                },
                "to": [
                    {
                        "email": recipient,
                    }
                ],
                "subject": subject,
                "textContent": body,
            }
            headers = {
                "api-key": BREVO_API_KEY,
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            resp = requests.post(
                "https://api.brevo.com/v3/smtp/email",
                headers=headers,
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            logging.info("Reminder email sent via Brevo to %s", recipient)
            return True

        # Fallback to classic SMTP settings (e.g. Gmail or Brevo SMTP relay)
        message = MIMEMultipart()
        message["From"] = EMAIL_FROM
        message["To"] = recipient
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain"))

        context = ssl.create_default_context()
        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT, timeout=30) as server:
            if EMAIL_USE_TLS:
                server.starttls(context=context)
            if EMAIL_USERNAME and EMAIL_PASSWORD:
                server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
            server.send_message(message)
        logging.info("Reminder email sent via SMTP to %s", recipient)
        return True
    except Exception as e:
        logging.error("Failed to send email to %s: %s", recipient, e)
        logging.debug(traceback.format_exc())
        return False


def compose_reminder_email(reminder, user_name, send_at=None):
    medication = reminder.get('medication', 'Medication')
    dosage = reminder.get('dosage', 'As directed')
    frequency = reminder.get('frequency', 'As prescribed')
    time_of_day = reminder.get('time_of_day', '-')

    lines = [
        f"Hi {user_name},",
        "",
    ]

    if send_at:
        try:
            scheduled_text = send_at.strftime('%b %d, %Y at %I:%M %p')
        except Exception:
            scheduled_text = str(send_at)
        lines.extend([
            f"Reminder scheduled for: {scheduled_text}",
            "",
        ])

    lines.extend([
        "This is a friendly reminder to take your prescribed medication:",
        f"• Medication: {medication}",
        f"• Dosage: {dosage}",
        f"• Frequency: {frequency}",
    ])

    if time_of_day:
        lines.append(f"• Scheduled time: {time_of_day}")

    if reminder.get('start_date_display'):
        lines.append(f"• Treatment started: {reminder['start_date_display']}")

    if reminder.get('end_date_display'):
        lines.append(f"• Treatment ends: {reminder['end_date_display']}")

    lines.extend([
        "",
        "Please follow your doctor's instructions.",
        "",
        "– MediScribe AI Reminder"
    ])

    return "Medication Reminder", "\n".join(lines)


def fetch_reminder_details(reminder_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        """
        SELECT r.id, r.user_id, r.medication, r.dosage, r.frequency, r.start_date,
               r.end_date, r.time_of_day, r.last_notification_sent,
               u.email_id, u.username
        FROM reminders r
        JOIN users u ON r.user_id = u.id
        WHERE r.id = ?
        """,
        (reminder_id,)
    )
    row = c.fetchone()
    conn.close()
    return row


def calculate_next_reminder_run(start_date_str, end_date_str, time_of_day_str, last_sent_iso=None):
    start_date = parse_date(start_date_str)
    reminder_time = parse_time_string(time_of_day_str)
    if not start_date or not reminder_time:
        return None

    end_date = parse_date(end_date_str) if end_date_str else None
    now = datetime.datetime.now(INDIA_TZ)

    last_sent = None
    if last_sent_iso:
        try:
            last_sent = datetime.datetime.fromisoformat(last_sent_iso)
            if last_sent.tzinfo is None:
                last_sent = last_sent.replace(tzinfo=INDIA_TZ)
            else:
                last_sent = last_sent.astimezone(INDIA_TZ)
        except ValueError:
            last_sent = None

    candidate_date = start_date
    if last_sent:
        candidate_date = max(candidate_date, last_sent.date())
    if candidate_date < now.date():
        candidate_date = now.date()

    while True:
        if end_date and candidate_date > end_date:
            return None

        due_datetime = datetime.datetime.combine(candidate_date, reminder_time, tzinfo=INDIA_TZ)

        if last_sent and due_datetime <= last_sent:
            candidate_date += datetime.timedelta(days=1)
            continue

        if due_datetime <= now:
            candidate_date += datetime.timedelta(days=1)
            continue

        if end_date and candidate_date > end_date:
            return None

        return due_datetime


def cancel_reminder_schedule(reminder_id):
    global scheduler
    if scheduler and scheduler.running:
        job_id = f"reminder_{reminder_id}"
        try:
            scheduler.remove_job(job_id)
        except Exception:
            pass


def schedule_reminder_email(reminder_id):
    global scheduler
    if not EMAIL_ENABLED:
        return

    if scheduler is None or not scheduler.running:
        start_reminder_scheduler(schedule_existing=False)

    row = fetch_reminder_details(reminder_id)
    if not row:
        return

    (
        _reminder_id,
        _user_id,
        medication,
        dosage,
        frequency,
        start_date,
        end_date,
        time_of_day,
        last_notification_sent,
        user_email,
        user_name,
    ) = row

    if not user_email:
        return

    next_run = calculate_next_reminder_run(start_date, end_date, time_of_day, last_notification_sent)
    if not next_run:
        cancel_reminder_schedule(reminder_id)
        return

    job_id = f"reminder_{reminder_id}"
    try:
        scheduler.remove_job(job_id)
    except Exception:
        pass

    scheduler.add_job(
        send_reminder_email_job,
        trigger='date',
        run_date=next_run,
        id=job_id,
        replace_existing=True,
        args=[reminder_id, next_run.isoformat()],
    )
    logging.info("Scheduled reminder %s for %s", reminder_id, next_run.isoformat())


def send_reminder_email_job(reminder_id, scheduled_iso):
    if not EMAIL_ENABLED:
        return

    try:
        scheduled_at = datetime.datetime.fromisoformat(scheduled_iso)
        if scheduled_at.tzinfo is None:
            scheduled_at = scheduled_at.replace(tzinfo=INDIA_TZ)
        else:
            scheduled_at = scheduled_at.astimezone(INDIA_TZ)
    except ValueError:
        scheduled_at = datetime.datetime.now(INDIA_TZ)

    row = fetch_reminder_details(reminder_id)
    if not row:
        return

    (
        _reminder_id,
        _user_id,
        medication,
        dosage,
        frequency,
        start_date,
        end_date,
        time_of_day,
        last_notification_sent,
        user_email,
        user_name,
    ) = row

    if not user_email:
        return

    reminder_payload = {
        'medication': medication,
        'dosage': dosage or 'As directed',
        'frequency': frequency,
        'time_of_day': time_of_day,
        'start_date_display': format_display_date(start_date),
        'end_date_display': format_display_date(end_date),
    }

    subject, body = compose_reminder_email(reminder_payload, user_name or 'there', scheduled_at)
    sent = send_email(subject, body, user_email)

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    if sent:
        c.execute(
            "UPDATE reminders SET last_notification_sent = ? WHERE id = ?",
            (scheduled_at.isoformat(), reminder_id),
        )
        conn.commit()
    conn.close()

    if sent:
        schedule_reminder_email(reminder_id)


def schedule_all_reminders():
    if not EMAIL_ENABLED:
        return
    if scheduler is None or not scheduler.running:
        return
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT id FROM reminders")
    reminder_ids = [row[0] for row in c.fetchall()]
    conn.close()

    for reminder_id in reminder_ids:
        schedule_reminder_email(reminder_id)


def check_due_reminders():
    now = datetime.datetime.now(INDIA_TZ)
    current_date = now.date()
    current_time = now.time()

    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute(
            """
            SELECT r.id, r.user_id, r.medication, r.dosage, r.frequency, r.start_date,
                   r.end_date, r.time_of_day, r.last_notification_sent,
                   u.email_id, u.username
            FROM reminders r
            JOIN users u ON r.user_id = u.id
            WHERE r.time_of_day IS NOT NULL AND r.time_of_day != ''
            """
        )
        rows = c.fetchall()

        for row in rows:
            (
                reminder_id,
                user_id,
                medication,
                dosage,
                frequency,
                start_date,
                end_date,
                time_of_day,
                last_notification_sent,
                user_email,
                user_name,
            ) = row

            if not user_email:
                continue

            start = parse_date(start_date)
            end = parse_date(end_date)
            reminder_time = parse_time_string(time_of_day)

            if reminder_time is None:
                continue
            if start and current_date < start:
                continue
            if end and current_date > end:
                continue

            due_datetime = datetime.datetime.combine(current_date, reminder_time, tzinfo=INDIA_TZ)
            if due_datetime > now:
                continue

            already_sent_today = False
            if last_notification_sent:
                try:
                    sent_dt = datetime.datetime.fromisoformat(last_notification_sent)
                    if sent_dt.tzinfo is None:
                        sent_dt = sent_dt.replace(tzinfo=INDIA_TZ)
                    else:
                        sent_dt = sent_dt.astimezone(INDIA_TZ)
                    already_sent_today = sent_dt.date() == current_date
                except ValueError:
                    already_sent_today = False

            if already_sent_today:
                continue

            reminder_payload = {
                'medication': medication,
                'dosage': dosage or 'As directed',
                'frequency': frequency,
                'time_of_day': time_of_day,
                'start_date_display': format_display_date(start_date),
                'end_date_display': format_display_date(end_date),
            }

            subject, body = compose_reminder_email(reminder_payload, user_name or 'there', due_datetime)
            sent = send_email(subject, body, user_email)

            if sent:
                c.execute(
                    "UPDATE reminders SET last_notification_sent = ? WHERE id = ?",
                    (due_datetime.isoformat(), reminder_id),
                )
                conn.commit()
                schedule_reminder_email(reminder_id)

        conn.close()
    except Exception as e:
        logging.error("Error while checking reminders: %s", e)
        logging.debug(traceback.format_exc())


def start_reminder_scheduler(schedule_existing=True):
    global scheduler
    if scheduler is not None and scheduler.running:
        return

    scheduler_instance = BackgroundScheduler()
    scheduler_instance.add_job(check_due_reminders, 'interval', minutes=1, id='reminder_checker', replace_existing=True)
    scheduler_instance.start()
    scheduler = scheduler_instance
    logging.info("Reminder scheduler started")

    if schedule_existing:
        schedule_all_reminders()

@app.route('/')
def home():
    """Root: redirect to landing (template home) or dashboard if logged in."""
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('landing'))


@app.route('/home')
def landing():
    """Template-based landing page with Services section (OCR module, etc.)."""
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        address = request.form['address']
        mobile_no = request.form['mobile_no']
        blood_group = request.form['blood_group']
        pin_code = request.form['pin_code']
        email_id = request.form['email_id']
        gender = request.form['gender']

        hashed_password = generate_password_hash(password)

        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password, address, mobile_no, blood_group, pin_code, email_id, gender) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                      (username, hashed_password, address, mobile_no, blood_group, pin_code, email_id, gender))
            conn.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists.', 'danger')
        finally:
            conn.close()
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()
        if user and check_password_hash(user[2], password):
            session['username'] = user[1]
            session['is_admin'] = user[3]
            session['user_id'] = user[0] # Store user_id in session
            flash('Login successful!', 'success')
            if user[3] == 1:
                return redirect(url_for('admin'))
            else:
                return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'success')
    return redirect(url_for('landing'))

@app.route('/dashboard')
def dashboard():
    if 'username' not in session or session.get('is_admin') == 1:
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session['username'])

def extract_contact_number(tags):
    if not tags:
        return None
    contact_keys = [
        'contact:phone', 'phone', 'contact:mobile', 'mobile',
        'contact:tel', 'tel'
    ]
    for key in contact_keys:
        value = tags.get(key)
        if value:
            return value
    return None


@app.route('/get_nearby_places')
def get_nearby_places():
    """Fetch nearby hospitals or pharmacies using OpenStreetMap Nominatim API"""
    lat = request.args.get('lat', type=float)
    lng = request.args.get('lng', type=float)
    place_type = request.args.get('type', 'hospital')  # 'hospital' or 'pharmacy'
    radius = request.args.get('radius', 5000, type=int)  # radius in meters
    
    if not lat or not lng:
        return jsonify({'error': 'Latitude and longitude are required'}), 400
    
    try:
        # Map place types to Nominatim search terms
        search_terms = {
            'hospital': 'hospital',
            'pharmacy': 'pharmacy'
        }
        
        search_term = search_terms.get(place_type, 'hospital')
        
        # Use Nominatim API to search for nearby places
        # Note: Nominatim has rate limits, so we use Overpass API for better results
        # Using Overpass Turbo API for more reliable results
        overpass_url = "http://overpass-api.de/api/interpreter"
        
        # Calculate bounding box (approximate)
        # 1 degree latitude ≈ 111 km, so for radius in meters:
        lat_offset = radius / 111000
        lng_offset = radius / (111000 * math.cos(math.radians(lat)))
        
        # Build Overpass QL query
        if place_type == 'hospital':
            query = f"""
            [out:json][timeout:25];
            (
              node["amenity"="hospital"]({lat - lat_offset},{lng - lng_offset},{lat + lat_offset},{lng + lng_offset});
              node["amenity"="clinic"]({lat - lat_offset},{lng - lng_offset},{lat + lat_offset},{lng + lng_offset});
              way["amenity"="hospital"]({lat - lat_offset},{lng - lng_offset},{lat + lat_offset},{lng + lng_offset});
              way["amenity"="clinic"]({lat - lat_offset},{lng - lng_offset},{lat + lat_offset},{lng + lng_offset});
            );
            out center meta;
            """
        else:  # pharmacy
            query = f"""
            [out:json][timeout:25];
            (
              node["amenity"="pharmacy"]({lat - lat_offset},{lng - lng_offset},{lat + lat_offset},{lng + lng_offset});
              node["shop"="pharmacy"]({lat - lat_offset},{lng - lng_offset},{lat + lat_offset},{lng + lng_offset});
              way["amenity"="pharmacy"]({lat - lat_offset},{lng - lng_offset},{lat + lat_offset},{lng + lng_offset});
              way["shop"="pharmacy"]({lat - lat_offset},{lng - lng_offset},{lat + lat_offset},{lng + lng_offset});
            );
            out center meta;
            """
        
        # Make request to Overpass API
        response = requests.post(overpass_url, data={'data': query}, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            places = []
            
            for element in data.get('elements', []):
                # Get coordinates
                if element.get('type') == 'node':
                    place_lat = element.get('lat')
                    place_lng = element.get('lon')
                elif element.get('type') == 'way':
                    center = element.get('center', {})
                    place_lat = center.get('lat')
                    place_lng = center.get('lon')
                else:
                    continue
                
                if not place_lat or not place_lng:
                    continue
                
                # Get name and address
                tags = element.get('tags', {})
                name = tags.get('name') or tags.get('name:en') or 'Unnamed ' + place_type.title()
                
                # Build address
                address_parts = []
                if tags.get('addr:street'):
                    address_parts.append(tags.get('addr:street'))
                if tags.get('addr:housenumber'):
                    address_parts.insert(0, tags.get('addr:housenumber'))
                if tags.get('addr:city'):
                    address_parts.append(tags.get('addr:city'))
                elif tags.get('addr:town'):
                    address_parts.append(tags.get('addr:town'))
                if tags.get('addr:postcode'):
                    address_parts.append(tags.get('addr:postcode'))
                
                address = ', '.join(address_parts) if address_parts else tags.get('addr:full') or 'Address not available'
                
                # Calculate distance
                distance = calculate_distance(lat, lng, place_lat, place_lng)
                
                places.append({
                    'name': name,
                    'lat': place_lat,
                    'lng': place_lng,
                    'address': address,
                    'distance': distance,
                    'phone': extract_contact_number(tags),
                    'type': place_type,
                })
            
            # Sort by distance
            places.sort(key=lambda x: x['distance'])
            
            return jsonify(places[:20])  # Return top 20 closest places
        
        else:
            # Fallback to Nominatim if Overpass fails
            return get_nearby_places_nominatim(lat, lng, place_type, radius)
            
    except requests.exceptions.RequestException as e:
        # Fallback to Nominatim if Overpass fails
        try:
            return get_nearby_places_nominatim(lat, lng, place_type, radius)
        except Exception as fallback_error:
            return jsonify({'error': str(fallback_error)}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_nearby_places_nominatim(lat, lng, place_type, radius):
    """Fallback method using Nominatim API"""
    search_terms = {
        'hospital': 'hospital',
        'pharmacy': 'pharmacy'
    }
    
    search_term = search_terms.get(place_type, 'hospital')
    
    # Use Nominatim search API
    nominatim_url = "https://nominatim.openstreetmap.org/search"
    params = {
        'q': search_term,
        'format': 'json',
        'limit': 20,
        'bounded': 1,
        'viewbox': f'{lng - 0.05},{lat + 0.05},{lng + 0.05},{lat - 0.05}',  # Approximate bounding box
        'addressdetails': 1
    }
    
    headers = {
        'User-Agent': 'MediScribe App'
    }
    
    response = requests.get(nominatim_url, params=params, headers=headers, timeout=10)
    
    if response.status_code == 200:
        data = response.json()
        places = []
        
        for item in data:
            place_lat = float(item.get('lat', 0))
            place_lng = float(item.get('lon', 0))
            
            if not place_lat or not place_lng:
                continue
            
            # Calculate distance
            distance = calculate_distance(lat, lng, place_lat, place_lng)
            
            # Check if within radius
            if distance * 1000 > radius:  # distance is in km, radius is in meters
                continue
            
            name = item.get('display_name', '').split(',')[0] or f'Unnamed {place_type.title()}'
            
            # Get address from display_name or address details
            address = item.get('display_name', 'Address not available')
            
            extra = item.get('extratags', {})
            phone = (
                item.get('phone')
                or extra.get('phone')
                or extra.get('contact:phone')
                or extra.get('contact:mobile')
            )

            places.append({
                'name': name,
                'lat': place_lat,
                'lng': place_lng,
                'address': address,
                'distance': distance,
                'phone': phone,
                'type': place_type,
            })
        
        places.sort(key=lambda x: x['distance'])
        return jsonify(places)
    else:
        return jsonify({'error': 'Failed to fetch places'}), 500

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in kilometers using Haversine formula"""
    R = 6371  # Radius of Earth in kilometers
    
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    
    return distance

@app.route('/admin')
def admin():
    if 'username' not in session or session.get('is_admin') != 1:
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('login'))
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT id, username, is_admin, address, mobile_no, blood_group, pin_code, email_id, gender FROM users")
    rows = c.fetchall()
    conn.close()

    users = []
    admin_count = 0
    for row in rows:
        raw_address = (row[3] or '').strip()
        raw_mobile = (row[4] or '').strip()
        raw_blood_group = (row[5] or '').strip()
        raw_pin_code = (row[6] or '').strip()
        raw_email = (row[7] or '').strip()
        raw_gender = (row[8] or '').strip()

        user_dict = {
            'id': row[0],
            'username': row[1],
            'is_admin': bool(row[2]),
            'role_label': 'Admin' if row[2] == 1 else 'User',
            'address': raw_address or 'Not provided',
            'mobile': raw_mobile or 'Not provided',
            'blood_group': raw_blood_group or 'Not set',
            'pin_code': raw_pin_code or 'Not set',
            'email': raw_email or 'Not provided',
            'gender': raw_gender or 'Prefer not to say',
        }
        user_dict['avatar'] = (user_dict['username'][:1] or '#').upper()

        normalized_mobile = re.sub(r'\D+', '', raw_mobile) if raw_mobile else ''

        search_fields = [
            user_dict['username'],
            raw_address,
            raw_mobile,
            normalized_mobile,
            raw_blood_group,
            raw_pin_code,
            raw_email,
            raw_gender,
            user_dict['role_label'],
        ]

        search_terms = []
        for field in search_fields:
            if not field:
                continue
            normalized = re.sub(r'[^a-z0-9]+', ' ', field.lower()).strip()
            if normalized:
                search_terms.append(normalized)

        user_dict['search_blob'] = ' '.join(search_terms)

        if user_dict['is_admin']:
            admin_count += 1

        users.append(user_dict)

    search_query_raw = request.args.get('query', '')
    normalized_query = re.sub(r'[^a-z0-9]+', ' ', search_query_raw.lower()).strip()
    search_tokens = [token for token in normalized_query.split() if token]

    if search_tokens:
        filtered_users = [
            user for user in users
            if all(token in user['search_blob'] for token in search_tokens)
        ]
    else:
        filtered_users = users

    total_users = len(users)
    regular_count = total_users - admin_count

    selected_role = request.args.get('role', 'all').lower()
    if selected_role not in ('all', 'admin', 'member'):
        selected_role = 'all'

    if selected_role in ('admin', 'member'):
        filtered_users = [
            user for user in filtered_users
            if (user['is_admin'] and selected_role == 'admin') or (not user['is_admin'] and selected_role == 'member')
        ]

    return render_template(
        'admin.html',
        users=filtered_users,
        total_users=total_users,
        admin_count=admin_count,
        regular_count=regular_count,
        filtered_count=len(filtered_users),
        search_query=search_query_raw,
        selected_role=selected_role,
    )


def _sanitize_threshold(value, default):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if numeric < 0.0:
        return 0.0
    if numeric > 1.0:
        return 1.0
    return numeric


@app.route('/admin/ocr-settings', methods=['POST'])
def admin_update_ocr_settings():
    if 'username' not in session or session.get('is_admin') != 1:
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('login'))

    current = load_ocr_settings()
    mode = (request.form.get('ocr_mode') or 'auto').lower()
    if mode not in ('auto', 'force_handwriting', 'force_tesseract'):
        mode = 'auto'

    updated = {
        'mode': mode,
        'quality_threshold': _sanitize_threshold(
            request.form.get('quality_threshold'),
            DEFAULT_OCR_SETTINGS['quality_threshold'],
        ),
        'medical_signal_threshold': _sanitize_threshold(
            request.form.get('medical_signal_threshold'),
            DEFAULT_OCR_SETTINGS['medical_signal_threshold'],
        ),
        'selection_margin': _sanitize_threshold(
            request.form.get('selection_margin'),
            DEFAULT_OCR_SETTINGS['selection_margin'],
        ),
        'min_score': _sanitize_threshold(
            request.form.get('min_score'),
            DEFAULT_OCR_SETTINGS['min_score'],
        ),
    }

    save_ocr_settings(updated)
    flash('OCR settings updated successfully.', 'success')
    return redirect(url_for('admin') + '#ocr-settings')


def load_user_history_results(target_user_id: int) -> list[dict]:
    results: list[dict] = []

    for filename in os.listdir(app.config['RESULTS_FOLDER']):
        if not filename.endswith('_results.json'):
            continue

        filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)

        try:
            with open(filepath, 'r') as f:
                result_data = json.load(f)
        except Exception as exc:
            logging.error("Error reading history file %s: %s", filename, exc)
            continue

        owner_id = result_data.get('user_id')
        try:
            owner_id_int = int(owner_id)
        except (TypeError, ValueError):
            owner_id_int = None

        if owner_id_int != target_user_id:
            continue

        timestamp_str = filename.split('_')[0]
        try:
            timestamp_value = int(timestamp_str)
        except ValueError:
            timestamp_value = int(time.time())

        result_data['timestamp'] = timestamp_value
        result_data['result_file'] = filename

        image_path = result_data.get('image_path')
        if image_path:
            result_data['image_filename'] = os.path.basename(image_path)

        preprocessed_path = result_data.get('preprocessed_image')
        if preprocessed_path:
            result_data['preprocessed_filename'] = os.path.basename(preprocessed_path)
            preprocessed_url = build_file_url(preprocessed_path)
            if preprocessed_url:
                result_data['preprocessed_url'] = preprocessed_url

        accuracy_info = result_data.get('accuracy')
        if isinstance(accuracy_info, (int, float)):
            overall_accuracy = float(accuracy_info)
            result_data['accuracy'] = {
                'overall_accuracy': overall_accuracy,
                'character_accuracy': overall_accuracy,
                'word_accuracy': overall_accuracy,
                'medication_accuracy': overall_accuracy,
            }
        elif not isinstance(accuracy_info, dict) or accuracy_info is None:
            result_data['accuracy'] = {
                'overall_accuracy': 'N/A',
                'character_accuracy': 'N/A',
                'word_accuracy': 'N/A',
                'medication_accuracy': 'N/A',
            }
        else:
            required_fields = ['overall_accuracy', 'character_accuracy', 'word_accuracy', 'medication_accuracy']
            for field in required_fields:
                if accuracy_info.get(field) is None:
                    accuracy_info[field] = 'N/A'
            result_data['accuracy'] = accuracy_info

        if 'medications' not in result_data:
            result_data['medications'] = []

        result_data['structured_prescriptions'] = build_structured_prescriptions(result_data)
        result_data['medication_count'] = len(result_data['structured_prescriptions'])

        raw_language_codes = (
            result_data.get('languages_used')
            or result_data.get('language_codes')
            or result_data.get('languages')
        )

        if isinstance(raw_language_codes, str):
            normalized_languages = normalize_language_codes([raw_language_codes])
        elif isinstance(raw_language_codes, list):
            normalized_languages = normalize_language_codes(raw_language_codes)
        else:
            normalized_languages = normalize_language_codes(None)

        result_data['languages_used'] = normalized_languages
        if not result_data.get('language_labels'):
            result_data['language_labels'] = get_language_labels(normalized_languages)

        results.append(result_data)

    results.sort(key=lambda item: item.get('timestamp', 0), reverse=True)
    return results


@app.route('/admin/delete-all-history', methods=['POST'])
def delete_all_history():
    if 'username' not in session or session.get('is_admin') != 1:
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('login'))

    deleted_records = 0
    had_errors = False

    for filename in os.listdir(app.config['RESULTS_FOLDER']):
        if not filename.endswith('_results.json'):
            continue

        filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)

        try:
            with open(filepath, 'r') as f:
                result_data = json.load(f)
        except Exception as exc:
            logging.error("Failed to read result file %s: %s", filename, exc)
            had_errors = True
            continue

        for key in ('image_path', 'preprocessed_image'):
            related_path = result_data.get(key)
            if related_path and os.path.exists(related_path):
                try:
                    os.remove(related_path)
                except Exception as exc:
                    logging.error("Failed to remove related file %s: %s", related_path, exc)
                    had_errors = True

        try:
            os.remove(filepath)
            deleted_records += 1
        except Exception as exc:
            logging.error("Failed to remove result file %s: %s", filepath, exc)
            had_errors = True

    if deleted_records:
        flash(f"Deleted {deleted_records} prescription record{'s' if deleted_records != 1 else ''}.", 'success')
    else:
        flash('No prescription history found to delete.', 'info')

    if had_errors:
        flash('Some files could not be deleted. Check server logs for details.', 'warning')

    return redirect(url_for('admin'))


@app.route('/admin/delete-user-history/<int:user_id>', methods=['POST'])
def delete_user_history(user_id: int):
    if 'username' not in session or session.get('is_admin') != 1:
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('login'))

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT username FROM users WHERE id = ?", (user_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        flash('User not found.', 'danger')
        return redirect(url_for('admin'))

    target_username = row[0]

    deleted_records = 0
    had_errors = False

    for filename in os.listdir(app.config['RESULTS_FOLDER']):
        if not filename.endswith('_results.json'):
            continue

        filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)

        try:
            with open(filepath, 'r') as f:
                result_data = json.load(f)
        except Exception as exc:
            logging.error("Failed to read result file %s: %s", filename, exc)
            had_errors = True
            continue

        owner_id = result_data.get('user_id')
        try:
            owner_id_int = int(owner_id)
        except (TypeError, ValueError):
            owner_id_int = None

        if owner_id_int != user_id:
            continue

        for key in ('image_path', 'preprocessed_image'):
            related_path = result_data.get(key)
            if related_path and os.path.exists(related_path):
                try:
                    os.remove(related_path)
                except Exception as exc:
                    logging.error("Failed to remove related file %s: %s", related_path, exc)
                    had_errors = True

        try:
            os.remove(filepath)
            deleted_records += 1
        except Exception as exc:
            logging.error("Failed to remove result file %s: %s", filepath, exc)
            had_errors = True

    if deleted_records:
        flash(
            f"Deleted {deleted_records} prescription record{'s' if deleted_records != 1 else ''} for {target_username}.",
            'success'
        )
    else:
        flash(f'No prescription history found for {target_username}.', 'info')

    if had_errors:
        flash('Some files could not be deleted. Check server logs for details.', 'warning')

    return redirect(url_for('admin'))


@app.route('/admin/user-history/<int:user_id>')
def admin_user_history(user_id: int):
    if 'username' not in session or session.get('is_admin') != 1:
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('login'))

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT id, username, email_id FROM users WHERE id = ?", (user_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        flash('User not found.', 'danger')
        return redirect(url_for('admin'))

    target_user = {
        'id': row[0],
        'username': row[1],
        'email': row[2] or 'Not provided',
    }

    results = load_user_history_results(user_id)

    return render_template(
        'history.html',
        results=results,
        admin_view=True,
        target_user=target_user,
        back_url=url_for('admin'),
    )


@app.route('/edit_user/<int:user_id>', methods=['GET', 'POST'])
def edit_user(user_id):
    if 'username' not in session or session.get('is_admin') != 1:
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('login'))

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    if request.method == 'POST':
        username = request.form['username']
        is_admin = 1 if 'is_admin' in request.form else 0
        password = request.form['password']
        address = request.form['address']
        mobile_no = request.form['mobile_no']
        blood_group = request.form['blood_group']
        pin_code = request.form['pin_code']
        email_id = request.form['email_id']
        gender = request.form['gender']

        if password:
            hashed_password = generate_password_hash(password)
            c.execute("UPDATE users SET username = ?, password = ?, is_admin = ?, address = ?, mobile_no = ?, blood_group = ?, pin_code = ?, email_id = ?, gender = ? WHERE id = ?",
                      (username, hashed_password, is_admin, address, mobile_no, blood_group, pin_code, email_id, gender, user_id))
        else:
            c.execute("UPDATE users SET username = ?, is_admin = ?, address = ?, mobile_no = ?, blood_group = ?, pin_code = ?, email_id = ?, gender = ? WHERE id = ?",
                      (username, is_admin, address, mobile_no, blood_group, pin_code, email_id, gender, user_id))
        conn.commit()
        conn.close()
        flash('User updated successfully.', 'success')
        return redirect(url_for('admin'))
    
    c.execute("SELECT id, username, is_admin, address, mobile_no, blood_group, pin_code, email_id, gender FROM users WHERE id = ?", (user_id,))
    user = c.fetchone()
    conn.close()

    if user is None:
        flash('User not found.', 'danger')
        return redirect(url_for('admin'))

    return render_template('edit_user.html', user=user)

@app.route('/delete_user/<int:user_id>', methods=['POST'])
def delete_user(user_id):
    if 'username' not in session or session.get('is_admin') != 1:
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('login'))
    
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()
    flash('User deleted successfully.', 'success')
    return redirect(url_for('admin'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)


@app.route('/set-language', methods=['POST'])
def set_language():
    language_code = request.form.get('language')
    if language_code not in UI_LANGUAGES:
        language_code = DEFAULT_UI_LANGUAGE

    session['preferred_language'] = language_code
    referrer = request.referrer
    return redirect(referrer or url_for('ocr_upload'))

def _upload_error(message, status_code=400):
    """Return an error response appropriate for HTML form or API usage."""
    if request.accept_mimetypes['application/json'] > request.accept_mimetypes['text/html']:
        return jsonify({'error': message}), status_code
    flash(message, 'danger')
    return redirect(url_for('ocr_upload'))


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if 'username' not in session:
        flash('Please login first.', 'danger')
        return redirect(url_for('login'))

    if request.method == 'GET':
        return redirect(url_for('ocr_upload'))

    if 'prescription_image' not in request.files:
        return _upload_error('Please choose a prescription image before submitting.')
    
    file = request.files.get('prescription_image')
    
    patient_name = request.form.get('patient_name', '')
    doctor_name = request.form.get('doctor_name', '')
    selected_language_codes = request.form.getlist('ocr_languages') or request.form.getlist('ocr_languages[]')
    language_codes = normalize_language_codes(selected_language_codes)
    
    if not patient_name or not doctor_name:
        return _upload_error('Patient name and doctor name are required.')
    
    if file is None or file.filename.strip() == '':
        return _upload_error('Please select a prescription image to upload.')
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        if not filename:
            ext = file.mimetype.split('/')[-1] if file.mimetype and '/' in file.mimetype else 'jpg'
            if ext == 'jpeg':
                ext = 'jpg'
            filename = f"prescription.{ext}"
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        try:
            results = process_prescription(
                filepath,
                app.config['RESULTS_FOLDER'],
                languages=language_codes,
            )
        except Exception as exc:
            logging.exception("Prescription processing failed for %s", filepath)
            return _upload_error(
                f"Failed to process prescription. Please try again with English selected, or a clearer photo. ({exc})"
            )

        if not isinstance(results, dict):
            return _upload_error('Processing returned an invalid response. Please try again.')

        if results.get('error') and not (results.get('raw_text') or '').strip():
            flash(f"OCR issue: {results['error']}", 'warning')
        
        if results.get('preprocessed_image'):
            preprocessed_url = build_file_url(results['preprocessed_image'])
            if preprocessed_url:
                results['preprocessed_url'] = preprocessed_url
        
        accuracy = evaluate_accuracy(results)
        results['accuracy'] = accuracy

        if results.get('is_trained') or results.get('trained_match'):
            structured_prescriptions = results.get('structured_prescriptions') or build_structured_prescriptions(results)
        else:
            structured_prescriptions = build_structured_prescriptions(results)
        results['structured_prescriptions'] = structured_prescriptions
        results['medication_count'] = len(structured_prescriptions)
        if structured_prescriptions and not results.get('medications'):
            results['medications'] = [med['name'] for med in structured_prescriptions if med.get('name')]

        results['patient_name'] = patient_name
        results['doctor_name'] = doctor_name
        results['date'] = datetime.datetime.now(INDIA_TZ).strftime('%Y-%m-%d')
        results['user_id'] = session.get('user_id') # Save user_id with results
        results['languages_used'] = results.get('languages_used') or language_codes
        results['language_labels'] = get_language_labels(results['languages_used'])
        
        results_filename = f"{timestamp}_results.json"
        results_path = os.path.join(app.config['RESULTS_FOLDER'], results_filename)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        return render_template('results.html', 
                               results=results, 
                               original_image=url_for('uploaded_file', filename=unique_filename))
    
    return _upload_error('Invalid file type. Please upload an image file (PNG, JPG, JPEG, WEBP, TIF, TIFF).')

@app.route('/ocr_upload')
def ocr_upload():
    if 'username' not in session:
        flash('Please login first.', 'danger')
        return redirect(url_for('login'))
    return render_template(
        'index.html',
        language_options=SUPPORTED_LANGUAGES,
        default_language_codes=DEFAULT_OCR_LANG_CODES,
        recommended_language_codes=RECOMMENDED_MULTILINGUAL_CODES,
        default_language_labels=get_language_labels(DEFAULT_OCR_LANG_CODES),
        recommended_language_labels=get_language_labels(RECOMMENDED_MULTILINGUAL_CODES),
    )

@app.route('/nearby')
def nearby():
    if 'username' not in session:
        flash('Please login first.', 'danger')
        return redirect(url_for('login'))
    return render_template('nearby.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/training')
def training():
    return render_template('training.html')

@app.route('/reminders')
def reminders():
    if 'username' not in session:
        flash('Please login first.', 'danger')
        return redirect(url_for('login'))

    user_id = session.get('user_id')
    if user_id is None:
        flash('User ID not found in session. Please login again.', 'danger')
        return redirect(url_for('login'))

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT id, medication, dosage, frequency, start_date, end_date, time_of_day, last_notification_sent FROM reminders WHERE user_id = ? ORDER BY start_date DESC", (user_id,))
    user_reminders = c.fetchall()
    conn.close()

    reminders_payload = []
    for reminder in user_reminders:
        (
            reminder_id,
            medication,
            dosage,
            frequency,
            start_date,
            end_date,
            time_of_day,
            last_notification_sent,
        ) = reminder

        start_display = format_display_date(start_date)
        end_display = format_display_date(end_date)
        last_sent_display = None
        if last_notification_sent:
            try:
                sent_dt = datetime.datetime.fromisoformat(last_notification_sent)
                last_sent_display = sent_dt.strftime('%b %d, %Y %I:%M %p')
            except ValueError:
                last_sent_display = last_notification_sent

        reminders_payload.append({
            'id': reminder_id,
            'medication': medication,
            'dosage': dosage or 'As directed',
            'frequency': frequency,
            'start_date': start_date,
            'end_date': end_date,
            'start_date_display': start_display or 'Start date not set',
            'end_date_display': end_display or 'Ongoing',
            'time_of_day': time_of_day or 'Anytime',
            'last_notification_sent': last_sent_display or 'Not sent yet',
        })

    return render_template('reminders.html', reminders=reminders_payload)

@app.route('/add_reminder', methods=['GET', 'POST'])
def add_reminder():
    if 'username' not in session:
        flash('Please login first.', 'danger')
        return redirect(url_for('login'))

    user_id = session.get('user_id')
    if user_id is None:
        flash('User ID not found in session. Please login again.', 'danger')
        return redirect(url_for('login'))

    if request.method == 'POST':
        medication = request.form['medication']
        dosage = request.form.get('dosage')
        frequency = request.form['frequency']
        start_date = request.form['start_date']
        end_date = request.form.get('end_date')
        time_of_day = request.form.get('time_of_day')

        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        reminder_id = None
        try:
            c.execute(
                "INSERT INTO reminders (user_id, medication, dosage, frequency, start_date, end_date, time_of_day, last_notification_sent) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (user_id, medication, dosage, frequency, start_date, end_date, time_of_day, None)
            )
            reminder_id = c.lastrowid
            conn.commit()
        except Exception as e:
            flash(f'Error adding reminder: {e}', 'danger')
        finally:
            conn.close()

        if reminder_id:
            schedule_reminder_email(reminder_id)
            flash('Reminder added successfully!', 'success')
            return redirect(url_for('reminders'))

    return render_template('add_reminder.html')

@app.route('/edit_reminder/<int:reminder_id>', methods=['GET', 'POST'])
def edit_reminder(reminder_id):
    if 'username' not in session:
        flash('Please login first.', 'danger')
        return redirect(url_for('login'))

    user_id = session.get('user_id')
    if user_id is None:
        flash('User ID not found in session. Please login again.', 'danger')
        return redirect(url_for('login'))

    if request.method == 'POST':
        medication = request.form['medication']
        dosage = request.form.get('dosage')
        frequency = request.form['frequency']
        start_date = request.form['start_date']
        end_date = request.form.get('end_date')
        time_of_day = request.form.get('time_of_day')

        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        success = False
        try:
            c.execute(
                "UPDATE reminders SET medication = ?, dosage = ?, frequency = ?, start_date = ?, end_date = ?, time_of_day = ?, last_notification_sent = NULL WHERE id = ? AND user_id = ?",
                (medication, dosage, frequency, start_date, end_date, time_of_day, reminder_id, user_id)
            )
            conn.commit()
            success = True
        except Exception as e:
            flash(f'Error updating reminder: {e}', 'danger')
        finally:
            conn.close()

        if success:
            schedule_reminder_email(reminder_id)
            flash('Reminder updated successfully!', 'success')
            return redirect(url_for('reminders'))

        return redirect(url_for('edit_reminder', reminder_id=reminder_id))

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT id, medication, dosage, frequency, start_date, end_date, time_of_day FROM reminders WHERE id = ? AND user_id = ?", (reminder_id, user_id))
    reminder = c.fetchone()
    conn.close()

    if reminder is None:
        flash('Reminder not found or you do not have permission to edit it.', 'danger')
        return redirect(url_for('reminders'))

    return render_template('edit_reminder.html', reminder=reminder)

@app.route('/delete_reminder/<int:reminder_id>', methods=['POST'])
def delete_reminder(reminder_id):
    if 'username' not in session:
        flash('Please login first.', 'danger')
        return redirect(url_for('login'))

    user_id = session.get('user_id')
    if user_id is None:
        flash('User ID not found in session. Please login again.', 'danger')
        return redirect(url_for('login'))

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        c.execute("DELETE FROM reminders WHERE id = ? AND user_id = ?", (reminder_id, user_id))
        conn.commit()
    except Exception as e:
        flash(f'Error deleting reminder: {e}', 'danger')
        conn.close()
        return redirect(url_for('reminders'))
    finally:
        conn.close()

    cancel_reminder_schedule(reminder_id)
    flash('Reminder deleted successfully!', 'success')
    return redirect(url_for('reminders'))

@app.route('/train', methods=['POST'])
def train_image():
    if 'username' not in session or not session.get('is_admin'):
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('login'))

    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['image']
    text_data = request.form.get('text', '')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        preferred_language = session.get('preferred_language')
        preferred_ocr_codes = None
        if preferred_language in UI_LANGUAGES:
            preferred_ocr_codes = UI_LANGUAGES[preferred_language].get('ocr_codes')

        results = process_prescription(
            filepath,
            app.config['RESULTS_FOLDER'],
            languages=preferred_ocr_codes,
        )
        
        text_data = (text_data or '').strip()
        if text_data:
            results['raw_text'] = text_data
            results['cleaned_text'] = text_data
        
        # Handle custom medication data from training form
        medicine_names = request.form.getlist('medicine_name[]')
        medicine_dosages = request.form.getlist('medicine_dosage[]')
        medicine_frequencies = request.form.getlist('medicine_frequency[]')
        
        # If custom medication data is provided, override the OCR results
        if medicine_names and any(name.strip() for name in medicine_names):
            # Build custom structured prescriptions
            custom_prescriptions = []
            for i in range(len(medicine_names)):
                if medicine_names[i].strip():  # Only add if name is not empty
                    custom_prescriptions.append({
                        'name': medicine_names[i].strip(),
                        'dosage': medicine_dosages[i].strip() if i < len(medicine_dosages) and medicine_dosages[i].strip() else 'As directed',
                        'frequency': medicine_frequencies[i].strip() if i < len(medicine_frequencies) and medicine_frequencies[i].strip() else 'As prescribed',
                        'duration': 'Until finished',
                        'instruction': 'Follow doctor instructions'
                    })
            
            # Update results with custom medication data
            results['structured_prescriptions'] = custom_prescriptions
            results['medication_count'] = len(custom_prescriptions)
            
            # Also update the medications arrays for backward compatibility
            results['medications'] = [med['name'] for med in custom_prescriptions]
            results['dosages'] = [med['dosage'] for med in custom_prescriptions]
            results['frequencies'] = [med['frequency'] for med in custom_prescriptions]
            results['routes'] = [med['instruction'] for med in custom_prescriptions]
            results['durations'] = [med['duration'] for med in custom_prescriptions]
        else:
            results['structured_prescriptions'] = build_structured_prescriptions(results)
            results['medication_count'] = len(results['structured_prescriptions'])

        from ocr_module.image_trainer import normalize_trained_results
        results = normalize_trained_results(results)
        results['confidence'] = 100.0

        trainer = ImageTrainer()
        success = trainer.add_training_sample(filepath, results)
        
        if success:
            results['patient_name'] = session.get('username', 'Training')
            results['doctor_name'] = 'Training Mode'
            results['date'] = datetime.datetime.now(INDIA_TZ).strftime('%Y-%m-%d')
            results['accuracy'] = evaluate_accuracy(results)
            flash('Image trained successfully. This is how it will appear on upload.', 'success')
            return render_template(
                'results.html',
                results=results,
                original_image=url_for('uploaded_file', filename=unique_filename),
            )
        else:
            return render_template('training_success.html', 
                                  error="Failed to train image", 
                                  error_message="There was a problem processing your image.")
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/history')
def history():
    if 'username' not in session:
        flash('Please login first.', 'danger')
        return redirect(url_for('login'))

    current_user_id = session.get('user_id')
    if current_user_id is None:
        flash('User ID not found in session. Please login again.', 'danger')
        return redirect(url_for('login'))

    results = load_user_history_results(current_user_id)

    return render_template('history.html', results=results)

@app.route('/result/<result_file>')
def view_result(result_file):
    if 'username' not in session:
        flash('Please login first.', 'danger')
        return redirect(url_for('login'))

    filepath = os.path.join(app.config['RESULTS_FOLDER'], result_file)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            results = json.load(f)
            
        # Apply the same accuracy normalization as in the history route
        if 'accuracy' in results:
            if isinstance(results['accuracy'], (int, float)):
                overall_accuracy = float(results['accuracy'])
                results['accuracy'] = {
                    'overall_accuracy': overall_accuracy,
                    'character_accuracy': overall_accuracy,
                    'word_accuracy': overall_accuracy,
                    'medication_accuracy': overall_accuracy
                }
            elif results['accuracy'] is None or not isinstance(results['accuracy'], dict):
                results['accuracy'] = {
                    'overall_accuracy': 'N/A',
                    'character_accuracy': 'N/A',
                    'word_accuracy': 'N/A',
                    'medication_accuracy': 'N/A'
                }
            elif isinstance(results['accuracy'], dict):
                required_fields = ['overall_accuracy', 'character_accuracy', 'word_accuracy', 'medication_accuracy']
                for field in required_fields:
                    if field not in results['accuracy'] or results['accuracy'][field] is None:
                        results['accuracy'][field] = 'N/A'
        else:
            results['accuracy'] = {
                'overall_accuracy': 'N/A',
                'character_accuracy': 'N/A',
                'word_accuracy': 'N/A',
                'medication_accuracy': 'N/A'
            }
            
        raw_language_codes = (
            results.get('languages_used')
            or results.get('language_codes')
            or results.get('languages')
        )

        if isinstance(raw_language_codes, str):
            normalized_languages = normalize_language_codes([raw_language_codes])
        elif isinstance(raw_language_codes, list):
            normalized_languages = normalize_language_codes(raw_language_codes)
        else:
            normalized_languages = normalize_language_codes(None)

        results['languages_used'] = normalized_languages
        if not results.get('language_labels'):
            results['language_labels'] = get_language_labels(normalized_languages)

        original_image = None
        if 'image_path' in results:
            original_image = os.path.basename(results['image_path'])
            
        preprocessed_image = None
        if results.get('preprocessed_image'):
            preprocessed_image = os.path.basename(results['preprocessed_image'])
            preprocessed_url = build_file_url(results['preprocessed_image'])
            if preprocessed_url:
                results['preprocessed_url'] = preprocessed_url
            
        results['structured_prescriptions'] = build_structured_prescriptions(results)
        results['medication_count'] = len(results['structured_prescriptions'])

        return render_template('results.html', 
                               results=results, 
                               original_image=url_for('uploaded_file', filename=original_image) if original_image else None)
    
    return "Result not found", 404

@app.route('/export-pdf/<result_file>')
def export_pdf(result_file):
    if 'username' not in session:
        flash('Please login first.', 'danger')
        return redirect(url_for('login'))

    filepath = os.path.join(app.config['RESULTS_FOLDER'], result_file)
    if not os.path.exists(filepath):
        return "Result not found", 404
        
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'MediScribe AI - Prescription Report', 0, 1, 'C')
    pdf.line(10, 20, 200, 20)
    pdf.ln(5)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Patient Information', 0, 1)
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 8, f"Patient: {results.get('patient_name', 'Unknown Patient')}", 0, 1)
    pdf.cell(0, 8, f"Doctor: {results.get('doctor_name', 'Unknown Doctor')}", 0, 1)
    pdf.cell(0, 8, f"Date: {results.get('date', datetime.datetime.now(INDIA_TZ).strftime('%Y-%m-%d'))}", 0, 1)
    pdf.ln(5)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Prescription Text', 0, 1)
    pdf.set_font('Arial', '', 11)
    
    text = results.get('raw_text', 'No text available')
    pdf.multi_cell(0, 8, text)
    pdf.ln(5)
    
    if 'medications' in results and results['medications']:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Medications Detected', 0, 1)
        pdf.set_font('Arial', '', 11)
        for medication in results['medications']:
            pdf.cell(0, 8, f"- {medication}", 0, 1)
    
    if 'accuracy' in results and results['accuracy']:
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Analysis Information', 0, 1)
        pdf.set_font('Arial', '', 11)
        pdf.cell(0, 8, f"Accuracy: {results['accuracy'].get('overall_accuracy', 'N/A')}%", 0, 1)
    
    patient_name = results.get('patient_name', 'Unknown')
    safe_patient_name = ''.join(c if c.isalnum() else '_' for c in patient_name)
    date_str = results.get('date', '').replace('-', '')
    filename = f"prescription_{safe_patient_name}_{date_str}.pdf"
    
    pdf_output = pdf.output(dest='S').encode('latin-1')
    response = make_response(pdf_output)
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    
    return response

@app.route('/export-all-csv')
def export_all_csv():
    if 'username' not in session:
        flash('Please login first.', 'danger')
        return redirect(url_for('login'))

    results_data = []
    for filename in os.listdir(app.config['RESULTS_FOLDER']):
        if filename.endswith('_results.json'):
            filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
            try:
                with open(filepath, 'r') as f:
                    result_data = json.load(f)
                    timestamp = filename.split('_')[0]
                    result_data['timestamp'] = int(timestamp)
                    result_data['result_file'] = filename
                    
                    if 'accuracy' in result_data:
                        if isinstance(result_data['accuracy'], (int, float)):
                            result_data['accuracy'] = {
                                'overall_accuracy': float(result_data['accuracy'])
                            }
                        elif not isinstance(result_data['accuracy'], dict):
                            result_data['accuracy'] = {
                                'overall_accuracy': 'N/A'
                            }
                    else:
                        result_data['accuracy'] = {
                            'overall_accuracy': 'N/A'
                        }
                    
                    results_data.append(result_data)
            except Exception as e:
                print(f"Error processing result file {filename} for CSV export: {e}")
    
    results_data.sort(key=lambda x: x['timestamp'], reverse=True)
    
    output = io.StringIO()
    csv_writer = csv.writer(output)
    
    csv_writer.writerow([
        'Date', 'Patient Name', 'Doctor Name', 'Accuracy', 
        'Medications', 'Raw Text'
    ])
    
    for result in results_data:
        date = datetime.datetime.fromtimestamp(result['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        patient_name = result.get('patient_name', 'Unknown Patient')
        doctor_name = result.get('doctor_name', 'Unknown Doctor')
        
        if isinstance(result.get('accuracy'), dict):
            accuracy = result['accuracy'].get('overall_accuracy', 'N/A')
        else:
            accuracy = 'N/A'
            
        medications = ', '.join(result.get('medications', [])) if 'medications' in result else 'None'
        raw_text = result.get('raw_text', 'No text available').replace('\n', ' ')
        
        csv_writer.writerow([date, patient_name, doctor_name, accuracy, medications, raw_text])
    
    response = make_response(output.getvalue())
    response.headers["Content-Type"] = "text/csv"
    response.headers["Content-Disposition"] = "attachment; filename=mediscribe_prescriptions.csv"
    
    return response

@app.route('/delete-record/<result_file>', methods=['POST'])
def delete_record(result_file):
    if 'username' not in session or not session.get('is_admin'):
        flash('Unauthorized access.', 'danger')
        return jsonify({'success': False, 'error': 'Unauthorized access.'}), 401

    filepath = os.path.join(app.config['RESULTS_FOLDER'], result_file)
    
    if not os.path.exists(filepath):
        return jsonify({'success': False, 'error': 'Record not found'}), 404
    
    try:
        with open(filepath, 'r') as f:
            result_data = json.load(f)
        
        if 'image_path' in result_data and result_data['image_path']:
            image_full_path = result_data['image_path']
            if os.path.exists(image_full_path):
                try:
                    os.remove(image_full_path)
                except Exception as e:
                    print(f"Error deleting original image file {image_full_path}: {e}")
        
        if 'preprocessed_image' in result_data and result_data['preprocessed_image']:
            preprocessed_full_path = result_data['preprocessed_image']
            if os.path.exists(preprocessed_full_path):
                try:
                    os.remove(preprocessed_full_path)
                except Exception as e:
                    print(f"Error deleting preprocessed image file {preprocessed_full_path}: {e}")
        
        os.remove(filepath)
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if EMAIL_ENABLED:
    if not app.debug or os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        start_reminder_scheduler()
else:
    logging.warning('Email notifications are disabled. Configure EMAIL_HOST/PORT/USERNAME/PASSWORD/EMAIL_FROM to enable reminder emails.')


if __name__ == '__main__':
    app.run(debug=True)
