import os
import cv2
import numpy as np
import json
import re
from PIL import Image
import pytesseract


# OCR quality thresholds
HANDWRITING_QUALITY_THRESHOLD = 0.70
PRINTED_TEXT_QUALITY_THRESHOLD = 0.85

import difflib
import spacy
from fuzzywuzzy import fuzz, process
from skimage import exposure, filters
from skimage.filters import unsharp_mask
from skimage.morphology import disk

try:
    # Try to load spacy model for medical NER
    nlp = spacy.load("en_core_sci_md")
except:
    try:
        # Fall back to standard English model
        nlp = spacy.load("en_core_web_sm")
    except:
        nlp = None
        print("Warning: Spacy model not loaded. Using fallback text processing.")

# Initialize Tesseract OCR - try PATH first, then common Windows install locations
def _configure_tesseract():
    import shutil
    tesseract_path = shutil.which('tesseract')
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        return
    common_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME', '')),
    ]
    for path in common_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return
    print("Warning: Tesseract not found. Please install Tesseract OCR.")

try:
    _configure_tesseract()
    version = pytesseract.get_tesseract_version()
    print(f'Tesseract ready ({version}) at {pytesseract.pytesseract.tesseract_cmd}')
except Exception as e:
    print(f"Warning: Tesseract verification failed: {e}")

DEFAULT_TESSERACT_LANG = "eng"
BASE_TESSERACT_CONFIGS = [
    ("--oem 3 --psm 6", 0.98),
    ("--oem 3 --psm 11", 0.92),
    ("--oem 1 --psm 4", 0.90),
    ("--oem 3 --psm 3", 0.85),
]


def normalize_tesseract_languages(language_codes):
    """Return a list of valid Tesseract language codes (defaults to English)."""
    if not language_codes:
        return [DEFAULT_TESSERACT_LANG]
    if isinstance(language_codes, str):
        language_codes = [language_codes]
    normalized = []
    seen = set()
    for raw in language_codes:
        if raw is None:
            continue
        code = str(raw).strip().lower()
        if code and code not in seen:
            normalized.append(code)
            seen.add(code)
    return normalized or [DEFAULT_TESSERACT_LANG]


def build_tesseract_config_list(language_codes=None):
    """Build weighted Tesseract config strings for the requested languages."""
    codes = normalize_tesseract_languages(language_codes)
    lang_option = f"-l {'+'.join(codes)}"
    return [(f"{base} {lang_option}", weight) for base, weight in BASE_TESSERACT_CONFIGS]
# Medical dictionary for common prescription terms
MEDICATION_DICT = {
    # Common medications - Updated for Indian market
    "amoxicillin": ["amox", "amoxil", "amoxicil", "amoxicilin", "mox", "novamox", "almoxi", "wymox"],
    "paracetamol": ["paracet", "parcetamol", "acetaminophen", "tylenol", "crocin", "panadol", "dolo", "metacin", "calpol", "sumo", "febrex", "acepar", "pacimol"],
    "ibuprofen": ["ibuprofin", "ibu", "ibuprofen", "advil", "motrin", "nurofen", "brufen", "ibugesic", "combiflam"],
    "aspirin": ["asa", "acetylsalicylic", "aspr", "disprin", "ecotrin", "bayer", "loprin", "delisprin", "colsprin"],
    "lisinopril": ["lisin", "prinivil", "zestril", "qbrelis", "listril", "hipril", "zestopril"],
    "metformin": ["metform", "glucophage", "fortamet", "glumetza", "riomet", "glycomet", "obimet", "gluformin", "glyciphage"],
    "atorvastatin": ["lipitor", "atorva", "atorvastat", "lipibec", "atorlip", "atocor", "storvas"],
    "levothyroxine": ["synthroid", "levothy", "levothyrox", "levoxyl", "tirosint", "euthyrox", "thyronorm", "eltroxin"],
    "omeprazole": ["prilosec", "omepraz", "losec", "zegerid", "priosec", "omez", "ocid", "prazole"],
    "amlodipine": ["norvasc", "amlo", "amlod", "katerzia", "norvasc", "amlopress", "amlopres", "amlokind"],
    "metoprolol": ["lopressor", "toprol", "metopro", "toprol-xl", "betaloc", "metolar", "starpress"],
    "sertraline": ["zoloft", "sert", "sertra", "lustral", "serta", "daxid", "serlin"],
    "gabapentin": ["neurontin", "gaba", "gabap", "gralise", "horizant", "gabapin", "gaban", "progaba"],
    "hydrochlorothiazide": ["hctz", "hydrochlor", "microzide", "hydrodiuril", "hydrazide", "aquazide"],
    "simvastatin": ["zocor", "simvast", "simlup", "simcard", "simvotin", "zosta", "simgal"],
    "losartan": ["cozaar", "losart", "lavestra", "repace", "losar", "zaart", "covance"],
    "albuterol": ["proventil", "ventolin", "proair", "salbutamol", "asthalin", "ventofort", "aeromist"],
    "fluoxetine": ["prozac", "sarafem", "rapiflux", "prodep", "fludac", "flunil", "flunat"],
    "citalopram": ["celexa", "cipramil", "citalo", "celepram", "citalex", "citopam"],
    "pantoprazole": ["protonix", "pantoloc", "pantocid", "pantodac", "zipant", "pan"],
    "furosemide": ["lasix", "furos", "frusemide", "frusol", "lasix", "frusenex", "diucontin"],
    "rosuvastatin": ["crestor", "rosuvast", "rosuvas", "rovista", "rostor", "colver"],
    "escitalopram": ["lexapro", "cipralex", "nexito", "feliz", "stalopam", "nexito-forte"],
    "montelukast": ["singulair", "montek", "montair", "monticope", "romilast", "monty-lc"],
    "prednisone": ["deltasone", "predni", "orasone", "predcip", "omnacortil", "wysolone"],
    "warfarin": ["coumadin", "jantoven", "warf", "warfex", "warfrant", "uniwarfin"],
    "tramadol": ["ultram", "tram", "tramahexal", "tramazac", "domadol", "tramacip"],
    "azithromycin": ["zithromax", "azithro", "z-pak", "azith", "azee", "aziwok", "azimax", "zithrocin"],
    "ciprofloxacin": ["cipro", "ciloxan", "ciproxin", "ciplox", "ciprobid", "cifran", "ciprinol"],
    "lamotrigine": ["lamictal", "lamot", "lamotrigin", "lamogard", "lamitor", "lametec"],
    "venlafaxine": ["effexor", "venlaf", "venlor", "veniz", "ventab", "venlift"],
    "insulin": ["lantus", "humulin", "novolin", "humalog", "novolog", "tresiba", "insugen", "wosulin", "basalog", "insuman", "apidra"],
    "metronidazole": ["flagyl", "metro", "metrogel", "metrogyl", "metrozole", "aristogyl"],
    "naproxen": ["aleve", "naprosyn", "anaprox", "xenar", "napra", "napxen"],
    "doxycycline": ["vibramycin", "oracea", "doxy", "doxin", "biodoxi", "doxt"],
    "cetirizine": ["zyrtec", "cetryn", "cetriz", "alerid", "cetcip", "zirtin", "cetzine"],
    "diazepam": ["valium", "valpam", "dizac", "calmpose", "zepose", "sedopam"],
    "alprazolam": ["xanax", "alprax", "tafil", "alp", "alzolam", "zolax", "restyl", "trika"],
    "clonazepam": ["klonopin", "rivotril", "clon", "petril", "clonopam", "lonazep"],
    "carvedilol": ["coreg", "carvedil", "cardivas", "carca", "carvil", "carloc"],
    "fexofenadine": ["allegra", "telfast", "fexofine", "fexova", "agimfast", "allerfast"],
    "ranitidine": ["zantac", "ranit", "rantec", "aciloc", "zinetac", "histac"],
    "diclofenac": ["voltaren", "diclof", "diclomax", "voveran", "diclonac", "reactin"],
    "ceftriaxone": ["rocephin", "ceftri", "cefaxone", "inocef", "trixone", "monotax"],
    "cefixime": ["suprax", "cefi", "taxim", "unice", "cefispan", "omnicef"],
    "esomeprazole": ["nexium", "esotrex", "esopral", "nexpro", "raciper", "sompraz"],
    "clopidogrel": ["plavix", "clopid", "plagerine", "clopilet", "deplatt", "noklot"],
    
    # Adding more Indian medications
    "levocetirizine": ["xyzal", "levocet", "teczine", "levazeo", "xyzra", "uvnil"],
    "febuxostat": ["uloric", "febugat", "febuget", "zylobact", "febustat"],
    "telmisartan": ["micardis", "telma", "telsar", "sartel", "telvas", "cresar"],
    "folic acid": ["folate", "folvite", "folet", "folacin", "folitab", "obifolic"],
    "olmesartan": ["benicar", "olmat", "olmy", "benitec", "olmezest", "olsar"],
    "vildagliptin": ["galvus", "zomelis", "jalra", "vysov", "vildalip", "viladay"],
    "sitagliptin": ["januvia", "sitagen", "istamet", "janumet", "sitaglip", "trevia"],
    "metoprolol": ["lopresor", "metolar", "metocard", "betaloc", "meto-er", "metopro"],
    "glimepiride": ["amaryl", "glimpid", "glymex", "zoryl", "glimer", "diaglip"],
    "gliclazide": ["diamicron", "lycazid", "glizid", "reclide", "odinase", "glynase"],
    "ramipril": ["altace", "cardiopril", "cardace", "ramiril", "celapres", "ramace"],
    "nebivolol": ["bystolic", "nebistar", "nebicard", "nebilong", "nebilet", "nubeta"],
    "cilnidipine": ["cilacar", "cinod", "ciladay", "neudipine", "cilaheart", "cidip"],
    "rabeprazole": ["aciphex", "rablet", "rabicip", "razo", "raboz", "pepcia"],
    "dexamethasone": ["decadron", "dexona", "dexamycin", "dexacort", "decilone", "dexasone"],
    "doxofylline": ["doxolin", "synasma", "doxobid", "doxovent", "doxoril", "doxfree"],
    "deflazacort": ["dezacor", "defcort", "defza", "xenocort", "flacort", "defolet"],
    "ondansetron": ["zofran", "ondem", "emeset", "vomitrol", "zondem", "ondemet"],
    "domperidone": ["motilium", "domstal", "vomistop", "dompan", "dompy", "domcolic"],
    "pantoprazole": ["pantocid", "pantop", "pantodac", "panto", "pan-d", "pantozol"],
    "mefenamic acid": ["ponstan", "meftal", "meflam", "mefkind", "rafen", "mefgesic"],
    "aceclofenac": ["aceclo", "hifenac", "zerodol", "movon", "aceclo-plus", "acebid"],
    "nimesulide": ["nise", "nimulid", "nimek", "nimica", "nimcet", "nimsaid"],
    "hydroxyzine": ["atarax", "anxnil", "hyzox", "hydryllin", "anxipar", "hydrax"],
    "amlodipine + atenolol": ["amlokind-at", "amtas-at", "stamlo-beta", "tenolam", "amlopress-at"],
    "telmisartan + hydrochlorothiazide": ["telma-h", "telsar-h", "telvas-h", "tazloc-h", "telista-h"],
    "sulfamethoxazole + trimethoprim": ["bactrim", "septran", "cotrim", "sepmax", "oriprim"],
    "amoxicillin + clavulanic acid": ["augmentin", "moxclav", "megaclox", "clavam", "hiclav", "clavum"],
    "ofloxacin": ["oflox", "oflin", "tarivid", "zenflox", "oflacin", "exocin"],
    "torsemide": ["demadex", "dytor", "tide", "torlactone", "presage", "tomide"],
    "chlorthalidone": ["thalitone", "clorpres", "cloress", "natrilix", "thaloride"],
    "ivermectin": ["stromectol", "ivermect", "ivecop", "ivepred", "scabo", "ivernex"],
    "rifaximin": ["xifaxan", "rifagut", "rcifax", "rifakem", "rifamide"],
    "nitrofurantoin": ["furadantin", "niftran", "nitrofur", "furadoine", "nidantin"],
    "betahistine": ["serc", "vertin", "betaserc", "vertigo", "beta", "histiwel"],
    "etizolam": ["etilaam", "etizola", "sedekopan", "etizaa", "etzee", "etova"],
    "clotrimazole": ["candid", "clotri", "mycomax", "candiderma", "candifun", "clotop"],
    "ketoconazole": ["nizoral", "sebizole", "ketoz", "fungicide", "ketomac", "ketostar"],
    "fluconazole": ["diflucan", "flucz", "forcan", "syscan", "zocon", "flucos"],
    "pregabalin": ["lyrica", "pregeb", "maxgalin", "nervalin", "pregastar", "pregica"],
    "methylprednisolone": ["medrol", "methylpred", "depo-medrol", "solu-medrol", "depopred", "medrate"],
    "levetiracetam": ["keppra", "levesam", "levroxa", "levipil", "levecetam", "epictal"],
    
    # Common dosage units
    "milligram": ["mg", "mgs", "millig", "milligram"],
    "microgram": ["mcg", "µg", "microg"],
    "gram": ["g", "gm", "gms", "gram"],
    "milliliter": ["ml", "mls", "millil"],
    
    # Common frequency terms
    "once daily": ["qd", "od", "daily", "once a day", "1 time a day", "1x day"],
    "twice daily": ["bid", "bd", "twice a day", "2 times a day", "2x day"],
    "three times daily": ["tid", "tds", "3 times a day", "3x day"],
    "four times daily": ["qid", "qds", "4 times a day", "4x day"],
    "every morning": ["qam", "morn", "morning"],
    "every night": ["qhs", "qpm", "noct", "night", "bedtime", "bed time"],
    "every hour": ["q1h", "hourly"],
    "every 4 hours": ["q4h", "4 hourly", "every 4 hrs"],
    "every 6 hours": ["q6h", "6 hourly", "every 6 hrs"],
    "every 8 hours": ["q8h", "8 hourly", "every 8 hrs"],
    "every 12 hours": ["q12h", "12 hourly", "every 12 hrs"],
    "as needed": ["prn", "pro re nata", "as required", "when necessary", "sos"],
    
    # Common routes of administration
    "by mouth": ["po", "oral", "orally", "per os"],
    "intravenous": ["iv", "i.v.", "ivp", "iv push"],
    "intramuscular": ["im", "i.m.", "intramuscul"],
    "subcutaneous": ["sc", "s.c.", "subq", "sub q", "subcu"],
    "sublingual": ["sl", "s.l.", "sublingual"],
    "topical": ["top", "topical", "externally"],
    "inhalation": ["inh", "inhale", "breathing"],
    
    # Common prescription instructions
    "with food": ["w/ food", "with meals", "with meal", "ac", "pc"],
    "before meals": ["ac", "a.c.", "before food"],
    "after meals": ["pc", "p.c.", "after food"],
    "with water": ["w/ water", "with h2o"],
    "do not crush": ["no crush", "donot crush", "do not chew", "swallow whole"],
    "take with plenty of water": ["take w/ plenty of h2o", "take w/ plenty of water"],
    "dissolve in water": ["dissolve", "dissolved in water"],
    "until finished": ["until gone", "to completion", "complete course"],
    "shake well": ["shake bottle", "mix well", "agitate"],
}

def apply_medical_dictionary_correction(text, medication_names=None):
    """Light spelling correction for known medicine names only (avoids false positives)."""
    if not text:
        return text
    if not medication_names:
        return text

    corrected_text = text
    for name in medication_names:
        tokens = [t for t in re.split(r'(\W+)', name) if t.strip() and re.search(r'\w', t)]
        for token in tokens:
            if len(token) < 4:
                continue
            best_match = None
            best_score = 0
            token_lower = token.lower()
            for key, aliases in MEDICATION_DICT.items():
                score = fuzz.ratio(token_lower, key.lower())
                if score > best_score and score >= 88:
                    best_score = score
                    best_match = key
                for alias in aliases:
                    score = fuzz.ratio(token_lower, alias.lower())
                    if score > best_score and score >= 88:
                        best_score = score
                        best_match = key
            if best_match:
                pattern = re.compile(re.escape(token), re.IGNORECASE)
                corrected_text = pattern.sub(best_match, corrected_text, count=1)
    return corrected_text


# Lines that are NOT bold medicine rows (composition / timing sub-lines)
_NON_MEDICINE_LINE_PREFIXES = (
    "composition", "compesition", "timing", "medicina", "medicine dosage",
    "nedicina", "complaints", "diagnosis", "reg.", "reg no",
)


def _is_non_medicine_line(text):
    lower = (text or "").strip().lower()
    if not lower:
        return True
    return any(lower.startswith(prefix) for prefix in _NON_MEDICINE_LINE_PREFIXES)


def _extract_strength_from_name(name):
    match = re.search(r"(\d+[\.\d]*\s*(?:MG|MCG|ML|G|GM))\b", name, re.IGNORECASE)
    if match:
        return re.sub(r"\s+", " ", match.group(1).upper())
    return None


def _parse_prescription_row_tail(tail):
    """Parse dosage / frequency / duration from the table columns after the medicine name."""
    dosage = None
    frequency = None
    duration = None
    if not tail:
        return dosage, frequency, duration

    dose_match = re.search(
        r"(\d+[\s\-=]+(?:\d+[\s\-=]+)+\d+|\d+\s*[\-–]\s*\d+[\s\-=]+\d+)",
        tail,
        re.IGNORECASE,
    )
    if dose_match:
        dosage = re.sub(r"\s+", "", dose_match.group(1).replace("=", "-"))

    if re.search(r"before\s+(?:food|breakfast)", tail, re.IGNORECASE):
        frequency = "Before food"
    elif re.search(r"after\s+(?:food|dinner|lunch|breakfast)", tail, re.IGNORECASE):
        frequency = "After food"

    if re.search(r"\b(?:daily|dally)\b", tail, re.IGNORECASE):
        frequency = f"{frequency} - Daily" if frequency else "Daily"

    duration_match = re.search(r"(\d+)\s*\.?\s*days?", tail, re.IGNORECASE)
    if duration_match:
        duration = f"{duration_match.group(1)} days"

    return dosage, frequency, duration


def extract_table_prescription_medicines(text):
    """
    Extract bold medicine names from clinic prescription tables.
    Format: 1) PAN HD * <dosage columns...>
    Skips Composition:/Timing: sub-lines (non-bold, smaller text).
    """
    if not text:
        return []

    medicines = []
    seen = set()

    def add_medicine(name, tail=""):
        name = re.sub(r"\s+", " ", (name or "").strip())
        name = re.sub(r"\s*\*\s*$", "", name).strip()
        if not name or _is_non_medicine_line(name):
            return
        key = name.lower()
        if key in seen:
            return
        seen.add(key)

        strength = _extract_strength_from_name(name)
        row_dosage, frequency, duration = _parse_prescription_row_tail(tail)
        dosage = strength or row_dosage or "As directed"

        medicines.append({
            "name": name,
            "dosage": dosage,
            "frequency": frequency or "As prescribed",
            "duration": duration or "Until finished",
            "instruction": "Follow doctor instructions",
        })

    row_pattern = re.compile(
        r"^\s*(\d+)\s*\)\s*(.+?)\s*\*\s*(.*)$",
        re.IGNORECASE,
    )
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = row_pattern.match(line)
        if match:
            add_medicine(match.group(2), match.group(3))
            continue
        # OCR sometimes misses the asterisk
        fallback = re.match(r"^\s*(\d+)\s*\)\s*(.+)$", line, re.IGNORECASE)
        if fallback:
            body = fallback.group(2).strip()
            if "*" in body:
                name_part, _, tail = body.partition("*")
                add_medicine(name_part, tail)
            elif re.search(r"\b(?:TAB|CAP|CAPSULE|SYRUP|MG|ML)\b", body, re.IGNORECASE):
                # Split before dosage schedule (digits with - or =)
                split = re.split(
                    r"\s+(?=\d+[\s\-=]+(?:\d+[\s\-=]+)+\d+|\d+\s*[\-–]\s*\d+)",
                    body,
                    maxsplit=1,
                )
                add_medicine(split[0], split[1] if len(split) > 1 else "")

    if medicines:
        print(f"Table prescription medicines ({len(medicines)}): {[m['name'] for m in medicines]}")
    return medicines


def entities_from_table_medicines(table_meds):
    """Convert table medicine rows to entity dict used by the OCR pipeline."""
    return {
        "medications": [m["name"] for m in table_meds],
        "dosages": [m["dosage"] for m in table_meds],
        "frequencies": [m["frequency"] for m in table_meds],
        "routes": [m["instruction"] for m in table_meds],
        "durations": [m["duration"] for m in table_meds],
        "structured_prescriptions": table_meds,
    }

# ---------------- NEW: Imports for additional OCR engines and preprocessing -------
try:
    import easyocr
except ImportError:
    easyocr = None
try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

import math

# ---------------- Modular Preprocessing Functions ------------
def advanced_preprocess_image(image_path, apply_clahe=True, resize_factor=1.5, deskew=True):
    img = cv2.imread(image_path)
    if img is None:
        return None
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # CLAHE for contrast
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=30)
    # Resize (upscale for small text)
    if resize_factor and resize_factor != 1.0:
        denoised = cv2.resize(denoised, (0, 0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_CUBIC)
    # Deskew
    if deskew:
        coords = np.column_stack(np.where(denoised > 0))
        angle = 0.0
        if coords.size > 0:
            rect = cv2.minAreaRect(coords)
            angle = rect[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            (h, w) = denoised.shape
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            denoised = cv2.warpAffine(denoised, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8)
    # Morphological opening/closing (try to make letters whole)
    kernel = np.ones((2,2), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    # Save enhanced
    base_name = os.path.basename(image_path)
    out_path = os.path.join(os.path.dirname(image_path), f'advanced_{base_name}')
    cv2.imwrite(out_path, processed)
    return {
        "original": image_path,
        "enhanced": out_path,
        "processed_image": processed
    }

# ------------- MULTI-BACKEND OCR LOGIC ---------------------
def extract_text_easyocr(image_path):
    if easyocr is None:
        print("EasyOCR not installed.")
        return ""
    reader = easyocr.Reader(['en'])
    try:
        results = reader.readtext(image_path)
        text = "\n".join([res[1] for res in results])
    except Exception as e:
        print(f"EasyOCR error: {e}")
        text = ""
    return text

def extract_text_paddleocr(image_path):
    if PaddleOCR is None:
        print("PaddleOCR not installed.")
        return ""
    ocr = PaddleOCR(lang='en', use_angle_cls=True, show_log=False)
    try:
        results = ocr.ocr(image_path)
        lines = []
        for line in results:
            for part in line:
                if len(part) > 1:
                    lines.append(part[1][0])
        return "\n".join(lines)
    except Exception as e:
        print(f"PaddleOCR error: {e}")
        return ""

def _select_preprocess_image(image_path, preprocess_mode="auto"):
    """Pick preprocessing pipeline. Basic enhancement works best for prescriptions."""
    if preprocess_mode == "basic":
        return preprocess_image(image_path)
    if preprocess_mode == "advanced":
        return advanced_preprocess_image(image_path)

    # auto: prefer basic; use advanced only if basic yields very little text
    basic = preprocess_image(image_path)
    if not basic:
        return advanced_preprocess_image(image_path)

    probe = run_multiple_ocr_passes(basic, max_configs=1)
    probe_text, _ = combine_ocr_results(probe)
    if len((probe_text or "").strip()) >= 20:
        return basic

    advanced = advanced_preprocess_image(image_path)
    if not advanced:
        return basic

    adv_probe = run_multiple_ocr_passes(advanced, max_configs=1)
    adv_text, _ = combine_ocr_results(adv_probe)
    if len((adv_text or "").strip()) > len((probe_text or "").strip()):
        return advanced
    return basic


# -------- MAIN ENTRY for MODULAR OCR (Tesseract/EasyOCR/PaddleOCR): --------------
def process_prescription_modular(
    image_path,
    output_dir=None,
    ocr_backend="tesseract",
    preprocess_mode="auto",
    languages=None,
):
    """
    ocr_backend: "tesseract" (default), "easyocr", "paddle"
    preprocess_mode: "auto" (choose best), "basic", "advanced"
    languages: list of Tesseract language codes (e.g. ["eng", "hin"])
    """
    language_codes = normalize_tesseract_languages(languages)
    image_data = _select_preprocess_image(image_path, preprocess_mode)
    if not image_data:
        return {
            "error": "Failed to preprocess image",
            "raw_text": "",
            "cleaned_text": "",
            "medications": [],
            "dosages": [],
            "frequencies": [],
            "routes": [],
            "confidence": 0.0,
            "languages_used": language_codes,
        }

    confidence = 0.0
    raw_text = ""
    if ocr_backend == "easyocr":
        raw_text = extract_text_easyocr(image_data["enhanced"])
        confidence = 85.0 if raw_text.strip() else 0.0
    elif ocr_backend == "paddle":
        raw_text = extract_text_paddleocr(image_data["enhanced"])
        confidence = 88.0 if raw_text.strip() else 0.0
    else:
        ocr_results = run_multiple_ocr_passes(image_data, language_codes=language_codes)
        raw_text, confidence = combine_ocr_results(ocr_results)
        confidence = float(confidence) * 100.0 if confidence <= 1.0 else float(confidence)

    if not raw_text:
        return {
            "image_path": image_path,
            "preprocessed_image": image_data.get("enhanced", ""),
            "raw_text": "",
            "cleaned_text": "",
            "medications": [],
            "dosages": [],
            "frequencies": [],
            "routes": [],
            "confidence": 0.0,
            "languages_used": language_codes,
        }

    table_meds = extract_table_prescription_medicines(raw_text)
    if table_meds:
        entities = entities_from_table_medicines(table_meds)
        corrected_text = raw_text
        extraction_mode = "prescription_table"
    else:
        entities = extract_medical_entities(raw_text)
        corrected_text = apply_medical_dictionary_correction(
            raw_text, entities.get("medications")
        )
        extraction_mode = "general"

    results = {
        "image_path": image_path,
        "preprocessed_image": image_data["enhanced"],
        "raw_text": raw_text,
        "cleaned_text": corrected_text,
        "medications": entities["medications"],
        "dosages": entities["dosages"],
        "frequencies": entities["frequencies"],
        "routes": entities["routes"],
        "durations": entities.get("durations", []),
        "structured_prescriptions": entities.get("structured_prescriptions", []),
        "medication_count": len(entities["medications"]),
        "confidence": confidence,
        "languages_used": language_codes,
        "ocr_engine": ocr_backend,
        "extraction_mode": extraction_mode,
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_no = os.path.splitext(os.path.basename(image_path))[0]
        out_json = os.path.join(output_dir, f"{base_no}_results.json")
        try:
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4)
        except Exception as exc:
            print(f"Could not save OCR results: {exc}")

    return results


# Backwards-compatible alias used by existing code paths
def process_prescription_with_enhanced_ocr(image_path, output_dir=None, languages=None):
    """Compatibility wrapper mapping to the modular implementation with defaults."""
    return process_prescription_modular(
        image_path,
        output_dir,
        ocr_backend="tesseract",
        preprocess_mode="auto",
        languages=languages,
    )

# ---------------- Google Lens-style OCR Ensemble -----------------
def _run_available_backends(image_path, image_data):
    texts = []
    # Tesseract (enhanced image with multi-psm passes)
    try:
        t_res = run_multiple_ocr_passes(image_data)
        t_text, t_conf = combine_ocr_results(t_res)
        if t_text:
            texts.append((t_text, float(t_conf)))
    except Exception as e:
        print(f"Tesseract pass error: {e}")
    # EasyOCR
    try:
        if easyocr is not None:
            e_text = extract_text_easyocr(image_data['enhanced'])
            if e_text:
                texts.append((e_text, 0.85))
    except Exception as e:
        print(f"EasyOCR ensemble error: {e}")
    # PaddleOCR
    try:
        if PaddleOCR is not None:
            p_text = extract_text_paddleocr(image_data['enhanced'])
            if p_text:
                texts.append((p_text, 0.88))
    except Exception as e:
        print(f"PaddleOCR ensemble error: {e}")
    return texts

def _merge_texts_google_style(text_with_scores):
    if not text_with_scores:
        return "", 0.0
    # Simple heuristic: pick the longest high-confidence text
    sorted_candidates = sorted(text_with_scores, key=lambda x: (len(x[0]), x[1]))
    best_text, best_score = sorted_candidates[-1]
    # Light cleanup: join broken hyphen lines, normalize spaces
    best_text = re.sub(r"\n\s*\n+", "\n", best_text)
    best_text = re.sub(r"[ \t]+", " ", best_text)
    return best_text.strip(), float(best_score)

def _extract_structured_items(text):
    # Use existing entity extractor and then enrich with per-line parsing
    entities = extract_medical_entities(text)
    medications = entities.get("medications", [])
    dosages = entities.get("dosages", [])
    frequencies = entities.get("frequencies", [])
    routes = entities.get("routes", [])
    # Try to map meds to nearby dosage/frequency by line proximity
    structured = []
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    for med in medications:
        best_line = None
        for line in lines:
            if re.search(r"\b" + re.escape(med) + r"\b", line, re.IGNORECASE):
                best_line = line
                break
        dose = None
        freq = None
        if best_line:
            m_dose = re.search(r"(\d+[\.\d]*\s*(?:mg|mcg|g|gm|ml|mL|units|tab(?:let)?s?|caps?|cap))", best_line, re.IGNORECASE)
            if m_dose:
                dose = m_dose.group(1)
            m_freq = re.search(r"(\d+\+\d+\+\d+|q\d+h|\b(?:od|bd|tds|qid|bid|tid)\b|once daily|twice daily|three times daily)", best_line, re.IGNORECASE)
            if m_freq:
                freq = m_freq.group(1)
        if dose is None:
            dose = next((d for d in dosages if re.search(r"\b" + re.escape(med) + r"\b", text, re.IGNORECASE)), None)
        if freq is None and frequencies:
            freq = frequencies[0]
        structured.append({
            "name": med,
            "dose": dose,
            "frequency": freq,
        })
    return structured, dosages, frequencies, routes

def process_prescription_google_style(image_path, output_dir=None):
    try:
        os.makedirs(output_dir or os.path.dirname(image_path), exist_ok=True)
        # Prefer advanced preprocessing; fall back to basic
        image_data = advanced_preprocess_image(image_path)
        if not image_data:
            image_data = preprocess_image(image_path)
        if not image_data:
            return {
                "error": "Failed to preprocess image",
                "raw_text": "",
                "cleaned_text": "",
                "medications": [],
                "dosages": [],
                "frequencies": [],
                "routes": []
            }
        # Ensemble of OCR engines (Lens-like behavior)
        texts = _run_available_backends(image_path, image_data)
        raw_text, conf = _merge_texts_google_style(texts)
        if not raw_text:
            return {
                "image_path": image_path,
                "preprocessed_image": image_data.get("enhanced", ""),
                "raw_text": "",
                "cleaned_text": "",
                "medications": [],
                "dosages": [],
                "frequencies": [],
                "routes": [],
                "confidence": 0.0,
            }
        corrected_text = apply_medical_dictionary_correction(raw_text)
        structured, dosages, frequencies, routes = _extract_structured_items(corrected_text)
        # Build results
        results = {
            "image_path": image_path,
            "preprocessed_image": image_data.get("enhanced", ""),
            "raw_text": raw_text,
            "cleaned_text": corrected_text,
            "medications": [s["name"] for s in structured] if structured else [],
            "dosages": list(set(dosages)),
            "frequencies": list(set(frequencies)),
            "routes": list(set(routes)),
            "items": structured,  # structured name/dose/frequency triplets
            "confidence": float(conf) * 100.0 if conf else 85.0,
        }
        # Optionally save JSON alongside
        if output_dir:
            base = os.path.basename(image_path)
            base_no = os.path.splitext(base)[0]
            out_json = os.path.join(output_dir, f"{base_no}_results.json")
            try:
                with open(out_json, 'w') as f:
                    json.dump(results, f, indent=4)
            except Exception as e:
                print(f"Could not save ensemble results: {e}")
        return results
    except Exception as e:
        print(f"Google-style OCR processing error: {e}")
        return {
            "error": f"Processing error: {str(e)}",
            "raw_text": "",
            "cleaned_text": "",
            "medications": [],
            "dosages": [],
            "frequencies": [],
            "routes": []
        }

def preprocess_image(image_path):
    """Simple and effective image preprocessing for prescription OCR."""
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        # Simple preprocessing steps for handwritten prescriptions
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray)
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.dilate(thresh, kernel, iterations=1)
        
        # Create filename for enhanced image
        base_name = os.path.basename(image_path)
        dir_name = os.path.dirname(image_path)
        enhanced_name = f"enhanced_{base_name}"
        enhanced_path = os.path.join(dir_name, enhanced_name)
        
        # Save enhanced image
        cv2.imwrite(enhanced_path, processed)
        
        return {
            "original": image_path,
            "enhanced": enhanced_path,
            "processed_image": processed
        }
    except Exception as e:
        print(f"Error in image preprocessing: {str(e)}")
        return None

def run_multiple_ocr_passes(image_data, language_codes=None, max_configs=None):
    """Run multiple OCR passes with different Tesseract configs and PSM modes."""
    try:
        results = []
        tess_configs = build_tesseract_config_list(language_codes)
        if max_configs:
            tess_configs = tess_configs[:max_configs]

        print("=== Starting OCR passes ===")

        # Try enhanced image with different configs
        if 'enhanced' in image_data and image_data['enhanced'] and os.path.exists(image_data['enhanced']):
            print(f"Trying enhanced image: {image_data['enhanced']}")
            for config, conf in tess_configs:
                result_text = extract_text_tesseract(image_data['enhanced'], config=config)
                if result_text and len(result_text.strip()) > 0:
                    print(f"Config [{config}]: extracted {len(result_text)} chars")
                    results.append((result_text, conf))
                    break

        # Try original image if enhanced failed
        if not results and 'original' in image_data and image_data['original'] and os.path.exists(image_data['original']):
            print(f"Trying original image: {image_data['original']}")
            for config, conf in tess_configs:
                result_text = extract_text_tesseract(image_data['original'], config=config)
                if result_text and len(result_text.strip()) > 0:
                    print(f"Config [{config}] on original: extracted {len(result_text)} chars")
                    results.append((result_text, conf * 0.9))
                    break
        
        # If we have no results yet, try different preprocessing on the image
        if not results and 'enhanced' in image_data and image_data['enhanced']:
            print("Trying inverted image...")
            enhanced_img = cv2.imread(image_data['enhanced'])
            if enhanced_img is not None:
                inverted = cv2.bitwise_not(enhanced_img)
                inverted_path = os.path.join(os.path.dirname(image_data['enhanced']), "inverted_temp.jpg")
                cv2.imwrite(inverted_path, inverted)
                result_text = extract_text_tesseract(inverted_path)
                if result_text and len(result_text.strip()) > 0:
                    print(f"Inverted image: Successfully extracted {len(result_text)} chars")
                    results.append((result_text, 0.85))
                else:
                    print("Inverted image: No text extracted")
                
                # Clean up
                try:
                    os.remove(inverted_path)
                except:
                    pass
        
        print(f"=== OCR passes complete: {len(results)} successful passes ===")
        
        # If still no results, return an empty placeholder result that won't break the system
        if not results:
            print("WARNING: All OCR passes failed. No text extracted.")
            return [("", 0.0)]
            
        return results
    
    except Exception as e:
        print(f"Error in OCR passes: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return [("", 0.0)]

def combine_ocr_results(results):
    """Combine text from multiple OCR passes, handling improved empty results"""
    try:
        if not results:
            return "", 0.0 # Return 0.0 confidence for no results
            
        all_text = []
        confidence_scores = []
        
        for text, confidence in results: # Iterate directly over (text, confidence) tuples
            if text:
                all_text.append(text)
                confidence_scores.append(float(confidence))
        
        if not all_text:
            return "", 0.0
            
        # Combine text from all passes, give preference to the first pass
        combined_text = all_text[0] if all_text else ""
        
        # Calculate average confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        return combined_text.strip(), avg_confidence
    
    except Exception as e:
        print(f"Error combining OCR results: {str(e)}")
        return "", 0.0

def extract_medical_entities(text):
    """Extract medical entities from the text with improved parsing"""
    medications = []
    dosages = []
    frequencies = []
    routes = []

    if not text or not text.strip():
        return {
            "medications": [],
            "dosages": [],
            "frequencies": [],
            "routes": [],
        }

    # Prefer numbered prescription table rows (bold medicine lines)
    table_meds = extract_table_prescription_medicines(text)
    if table_meds:
        return entities_from_table_medicines(table_meds)

    # Split text into lines for better parsing
    lines = text.split('\n')
    
    # Use spaCy for entity recognition if available
    if nlp:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ["CHEMICAL", "DRUG", "MEDICATION"]:
                med_name = re.sub(r'\s+\d+\s*\w*\b', '', ent.text)
                med_name = re.sub(r'\b(once|twice|three|four)(\s+times)?\s+(daily|a\s+day)\b', '', med_name, flags=re.IGNORECASE)
                med_name = re.sub(r'\b(every|each)\s+(morning|evening|night|day|hour|hourly)\b', '', med_name, flags=re.IGNORECASE)
                med_name = re.sub(r'\b(qd|bid|tid|qid|prn|od|q\d+h)\b', '', med_name, flags=re.IGNORECASE)
                med_name = med_name.strip()
                if med_name and len(med_name) > 2:
                    medications.append(med_name)
    
    # Enhanced medicine name extraction - look for capitalized words/phrases that might be medicine names
    # This helps find medicines not in dictionary
    skip_words = [
        'Patient', 'Doctor', 'Date', 'Name', 'Address', 'Phone',
        'Prescription', 'Rx', 'Take', 'Before', 'After', 'Morning',
        'Evening', 'Night', 'Food', 'Water', 'Day', 'Week', 'Month',
    ]

    for line in lines:
        line = line.strip()
        if not line or len(line) < 3:
            continue
        
        # Pattern: Capitalized word(s) followed by numbers/units (likely medicine)
        # Examples: "Paracetamol 500mg", "Amoxicillin 250mg", "Crocin 500"
        med_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:\d+[\.\d]*\s*(?:mg|mcg|g|ml|tablet|capsule|tab|cap))'
        matches = re.finditer(med_pattern, line)
        for match in matches:
            med_name = match.group(1).strip()
            if med_name not in skip_words and len(med_name) > 2:
                medications.append(med_name)
        
        # Pattern for medicine names with dashes/slashes (e.g., "Linets 5/25", "Losuco-50")
        dash_med_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*[-/]\s*\d+'
        matches = re.finditer(dash_med_pattern, line)
        for match in matches:
            med_name = match.group(1).strip()
            if med_name not in skip_words and len(med_name) > 2:
                medications.append(med_name)
        
        # Pattern: Medicine name followed by dosage pattern (e.g., "Medicine 1 tablet")
        simple_med_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+\d+\s*(?:tablet|cap|mg|mcg|ml|g)\b'
        matches = re.finditer(simple_med_pattern, line, re.IGNORECASE)
        for match in matches:
            med_name = match.group(1).strip()
            if med_name not in skip_words and len(med_name) > 2:
                medications.append(med_name)
    
    # Pattern for dosage (number + unit)
    dosage_pattern = r'\b(\d+[\.\d]*)\s*(mg|mcg|mL|ml|g|gm|mg/mL|mEq|units|tablets?|caps?|tab|cap)\b'
    dosage_matches = re.finditer(dosage_pattern, text, re.IGNORECASE)
    for match in dosage_matches:
        dosages.append(match.group(0))
    
    # Enhanced frequency patterns (including Indian format like 1+0+1)
    freq_patterns = [
        r'\b(once|twice|three times|four times)\s+daily\b',
        r'\b(q\.?d|b\.?i\.?d|t\.?i\.?d|q\.?i\.?d|od|bd|tds|qid)\b',
        r'\b(every|each)\s+(\d+)\s+(hours?|days?)\b',
        r'\b(q)(\d+)(h)\b',
        r'\bprn\b',
        r'\bas needed\b',
        r'\b\d+\+\d+\+\d+\b',  # Pattern like 1+0+1 (morning+afternoon+night)
        r'\b\d+\s*x\s*\d+\b',   # Pattern like "2x3" (2 times, 3 days)
        r'\b\d+\s+times?\s+(daily|a\s+day)\b'
    ]
    
    for pattern in freq_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            freq_text = match.group(0)
            if freq_text not in frequencies:
                frequencies.append(freq_text)
    
    # Pattern for routes of administration
    route_patterns = [
        r'\b(oral(ly)?|by mouth|p\.?o\.)\b',
        r'\b(intravenous|i\.?v\.)\b',
        r'\b(intramuscular|i\.?m\.)\b',
        r'\b(subcutaneous|s\.?c\.|sub-q)\b',
        r'\b(topical(ly)?)\b',
        r'\b(sublingual|s\.?l\.)\b'
    ]
    
    for pattern in route_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            routes.append(match.group(0))
    
    # Clean medication names to remove any residual dosage or frequency info
    clean_medications = []
    seen = set()
    for med in medications:
        # Remove dosage and frequency information
        clean_med = re.sub(r'\s+\d+\s*\w*\b', '', med).strip()
        clean_med = re.sub(r'\b(once|twice|three|four)(\s+times)?\s+(daily|a\s+day)\b', '', clean_med, flags=re.IGNORECASE).strip()
        clean_med = re.sub(r'\b(every|each)\s+(morning|evening|night|day|hour|hourly)\b', '', clean_med, flags=re.IGNORECASE).strip()
        clean_med = re.sub(r'\b(qd|bid|tid|qid|prn|od|q\d+h)\b', '', clean_med, flags=re.IGNORECASE).strip()
        
        if clean_med and len(clean_med) > 2:
            clean_med_lower = clean_med.lower()
            if clean_med_lower not in seen:
                seen.add(clean_med_lower)
                clean_medications.append(clean_med)
    
    # Debug: Print extracted medicines for troubleshooting
    if clean_medications:
        print(f"Extracted {len(clean_medications)} medications: {clean_medications}")
    else:
        print("No medications extracted from text. Raw text preview:", text[:200])
    
    return {
        "medications": clean_medications,
        "dosages": list(set(dosages)),
        "frequencies": list(set(frequencies)),
        "routes": list(set(routes))
        }

# ===================================================
# Text Extraction using Tesseract OCR
# ===================================================
def extract_text_tesseract(image, config=None):
    """Extract text using Pytesseract with improved error handling"""
    try:
        # Test Tesseract availability first
        try:
            pytesseract.get_tesseract_version()
        except Exception as te:
            print(f"ERROR: Tesseract not found or not accessible: {te}")
            print(f"Tesseract path being used: {pytesseract.pytesseract.tesseract_cmd}")
            return ""
        
        # Use optimized OCR config for better results
        if config is None:
            # Try PSM 6 first (single uniform block), if fails try PSM 11 (sparse text) or PSM 3 (auto)
            config = '--psm 6 --oem 3'
            # PSM 6: Assume a single uniform block of text
            # PSM 11: Sparse text (good for prescriptions)
            # OEM 3: Default OCR Engine Mode
        
        text = ""
        
        # Pytesseract works well with PIL Images or file paths
        if isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            if len(image.shape) == 3:
                # Color image
                image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                # Grayscale image
                image_pil = Image.fromarray(image)
            
            print(f"Extracting text from numpy array image, shape: {image.shape}")
            text = pytesseract.image_to_string(image_pil, config=config, lang='eng')
            
        elif isinstance(image, str) and os.path.exists(image):
            print(f"Extracting text from file: {image}")
            # Read image and convert to PIL if needed
            img = cv2.imread(image)
            if img is not None:
                # Try with original image
                image_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                text = pytesseract.image_to_string(image_pil, config=config, lang='eng')
                print(f"Text extracted (length: {len(text)}): {text[:100]}...")
            else:
                print(f"ERROR: Could not read image file: {image}")
                return ""
        else:
            print(f"Unsupported image format. Type: {type(image)}, Path exists: {os.path.exists(image) if isinstance(image, str) else 'N/A'}")
            return ""
        
        extracted_text = text.strip()
        print(f"Successfully extracted {len(extracted_text)} characters from image")
        return extracted_text
        
    except pytesseract.TesseractNotFoundError:
        print("ERROR: Tesseract executable not found. Please install Tesseract OCR.")
        print(f"Expected path: {pytesseract.pytesseract.tesseract_cmd}")
        return ""
    except Exception as e:
        print(f"Pytesseract error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return ""

# If running as a script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Prescription OCR with Pytesseract")
    parser.add_argument("--image", "-i", required=True, help="Path to prescription image")
    parser.add_argument("--output", "-o", default="./output", help="Output directory for results")
    
    args = parser.parse_args()
    
    print("===== Enhanced Prescription OCR Processing =====")
    print(f"Processing image: {args.image}")
    
    # Process the prescription
    results = process_prescription_modular(args.image, args.output)
    
    # Print the results
    if "error" in results:
        print(f"Error: {results['error']}")
    else:
        print("\n===== OCR Results =====")
        print(f"Preprocessed image: {results['preprocessed_image']}")
        
        print("\nExtracted text:")
        print(results['raw_text'])
        
        print("\nCleaned text:")
        print(results['cleaned_text'])
        
        print("\nDetected medications:")
        if results['medications']:
            for med in results['medications']:
                print(f"- {med}")
        else:
            print("No medications detected")
            
        print("\nDetected dosages:")
        if results['dosages']:
            for dosage in results['dosages']:
                print(f"- {dosage}")
        else:
            print("No dosages detected")
            
        print("\nDetected frequencies:")
        if results['frequencies']:
            for freq in results['frequencies']:
                print(f"- {freq}")
        else:
            print("No frequencies detected")
            
        print("\nDetected routes:")
        if results['routes']:
            for route in results['routes']:
                print(f"- {route}")
        else:
            print("No routes detected")
        
        print(f"\nConfidence: {results.get('confidence', 0):.1f}%")
        
        print("\nProcessing complete!")
