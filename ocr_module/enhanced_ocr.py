"""
Enhanced OCR module for prescription text extraction.

Improvements:
- Image preprocessing: grayscale, noise removal (fastNlMeans + optional bilateral),
  Gaussian blur, adaptive/Otsu thresholding, resizing for accuracy, morphology.
- Pytesseract: --oem 3 --psm 6 (and fallback PSM modes), multi-language support.
- Error handling and logging for production debugging.
- EasyOCR/PaddleOCR used for handwriting and when Tesseract output is poor.

Handwritten text: Use EasyOCR or PaddleOCR (force_handwriting / force_paddleocr).
Low-resolution: Preprocessing upscales small images and applies sharpening;
  for very low-res, consider super-resolution (e.g. OpenCV DNN or ESRGAN) then OCR.
"""
import os
import logging
import cv2
import numpy as np
import json
import re
from PIL import Image
import pytesseract  # type: ignore
import difflib
from fuzzywuzzy import fuzz, process
from skimage import exposure, filters
from skimage.filters import unsharp_mask
from skimage.morphology import disk

# Production-ready logging: level INFO for normal runs, DEBUG for troubleshooting
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

_easyocr_import_error = None
try:
    import easyocr  # type: ignore
except Exception as e:
    easyocr = None
    _easyocr_import_error = e

_paddleocr_import_error = None
try:
    from paddleocr import PaddleOCR  # type: ignore
    PADDLEOCR_AVAILABLE = True
except Exception as e:
    PaddleOCR = None
    PADDLEOCR_AVAILABLE = False
    _paddleocr_import_error = e

try:
    import torch
    EASY_OCR_USE_GPU = torch.cuda.is_available()
    PADDLEOCR_USE_GPU = torch.cuda.is_available() if PADDLEOCR_AVAILABLE else False
except Exception:
    EASY_OCR_USE_GPU = False
    PADDLEOCR_USE_GPU = False

# Log OCR engine availability at import (helps debug "no text" issues)
def _log_ocr_availability():
    if easyocr is None:
        logger.warning("EasyOCR not loaded. Handwriting OCR will be skipped. %s", _easyocr_import_error or "Install: pip install easyocr")
    else:
        logger.info("EasyOCR available for handwriting OCR.")
    if not PADDLEOCR_AVAILABLE and _paddleocr_import_error:
        logger.debug("PaddleOCR not loaded: %s", _paddleocr_import_error)
_log_ocr_availability()

# Try to import and load spacy model for medical NER (optional)
try:
    import spacy
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
except Exception as e:
    nlp = None
    print(f"Warning: Spacy not available ({e}). Using fallback text processing.")

# Initialize Tesseract OCR (override via env TESSERACT_CMD if needed)
_tesseract_cmd = os.environ.get("TESSERACT_CMD")
if _tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd
else:
    # Default Windows path; Linux/Mac often find tesseract in PATH
    _default_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        '/usr/bin/tesseract',
        '/usr/local/bin/tesseract',
    ]
    for _p in _default_paths:
        if os.path.isfile(_p):
            pytesseract.pytesseract.tesseract_cmd = _p
            break
    else:
        logger.warning("Tesseract executable not found in default paths; ensure it is in PATH.")

DEFAULT_LANGUAGE_CODES = ["eng"]
BASE_TESSERACT_CONFIGS = [
    ("--oem 3 --psm 6", 0.98),
    ("--oem 3 --psm 11", 0.92),
    ("--oem 1 --psm 4", 0.9),
    ("--oem 3 --psm 5", 0.88),
]

HANDWRITING_QUALITY_THRESHOLD = 0.6
HANDWRITING_SELECTION_MARGIN = 0.05
HANDWRITING_MIN_SCORE = 0.45
MEDICAL_SIGNAL_THRESHOLD = 0.08
OCR_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ocr_config.json')
DEFAULT_RUNTIME_OCR_SETTINGS = {
    'mode': 'auto',
    'quality_threshold': HANDWRITING_QUALITY_THRESHOLD,
    'medical_signal_threshold': MEDICAL_SIGNAL_THRESHOLD,
    'selection_margin': HANDWRITING_SELECTION_MARGIN,
    'min_score': HANDWRITING_MIN_SCORE,
}
_runtime_settings_cache = None
_runtime_settings_mtime = None

EASYOCR_LANGUAGE_MAP = {
    "eng": "en",
    "hin": "hi",
    "mar": "mr",
    "kan": "kn",
    "tam": "ta",
    "tel": "te",
    "mal": "ml",
    "guj": "gu",
    "ben": "bn",
}

_easyocr_readers = {}
_paddleocr_reader = None


def _load_runtime_settings():
    global _runtime_settings_cache, _runtime_settings_mtime
    try:
        mtime = os.path.getmtime(OCR_CONFIG_PATH)
    except OSError:
        mtime = None

    if _runtime_settings_cache is not None and _runtime_settings_mtime == mtime:
        return _runtime_settings_cache

    data = {}
    if mtime is not None:
        try:
            with open(OCR_CONFIG_PATH, 'r') as f:
                file_data = json.load(f)
            if isinstance(file_data, dict):
                data = file_data
        except Exception as exc:
            print(f"OCR settings load error: {exc}")

    merged = DEFAULT_RUNTIME_OCR_SETTINGS.copy()
    for key in merged:
        if key in data:
            merged[key] = data[key]

    _runtime_settings_cache = merged
    _runtime_settings_mtime = mtime
    return merged


def _sanitize_language_codes(language_codes=None):
    if language_codes is None:
        return DEFAULT_LANGUAGE_CODES.copy()

    if isinstance(language_codes, str):
        language_codes = [language_codes]

    cleaned = []
    seen = set()
    for code in language_codes:
        normalized = (code or "").strip().lower()
        if not normalized:
            continue
        if normalized not in seen:
            cleaned.append(normalized)
            seen.add(normalized)

    return cleaned or DEFAULT_LANGUAGE_CODES.copy()


def build_tesseract_configs(language_codes=None):
    codes = _sanitize_language_codes(language_codes)
    lang_option = f"-l {'+'.join(codes)}"
    return [(f"{base} {lang_option}", weight) for base, weight in BASE_TESSERACT_CONFIGS]


def _map_languages_for_easyocr(language_codes=None):
    codes = _sanitize_language_codes(language_codes)
    mapped = []
    for code in codes:
        easy_code = EASYOCR_LANGUAGE_MAP.get(code)
        if easy_code and easy_code not in mapped:
            mapped.append(easy_code)
    if not mapped:
        mapped.append("en")
    return mapped


def _get_easyocr_reader(language_codes):
    if easyocr is None:
        return None
    key = tuple(language_codes)
    if key in _easyocr_readers:
        return _easyocr_readers[key]
    try:
        # Use gpu=False to avoid CUDA/GPU issues; set to EASY_OCR_USE_GPU if you have a working GPU
        reader = easyocr.Reader(language_codes, gpu=False, verbose=False)
        _easyocr_readers[key] = reader
        return reader
    except Exception as e:
        logger.warning("EasyOCR Reader creation failed for %s: %s", language_codes, e)
        return None


def run_handwriting_ocr(image_path, language_codes=None):
    """Extract text using EasyOCR (good for handwritten and mixed content)."""
    if easyocr is None:
        logger.warning("EasyOCR not available; install with: pip install easyocr")
        return "", 0.0

    mapped_codes = _map_languages_for_easyocr(language_codes)
    # Load image: numpy array avoids path/encoding issues on Windows
    img = None
    if isinstance(image_path, str) and os.path.isfile(image_path):
        img = cv2.imread(image_path)
        if img is None:
            try:
                from PIL import Image as PILImage
                pil_img = PILImage.open(image_path).convert("RGB")
                img = np.array(pil_img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            except Exception as e:
                logger.warning("EasyOCR: could not load image (cv2 and PIL failed): %s", e)
        else:
            logger.debug("EasyOCR: loaded image %s shape %s", image_path, img.shape)
    elif isinstance(image_path, np.ndarray):
        img = image_path

    if img is None:
        logger.warning("EasyOCR: could not load image from %s", image_path)
        return "", 0.0

    try:
        reader = _get_easyocr_reader(mapped_codes)
        if reader is None:
            logger.warning("EasyOCR: Reader not available for %s", mapped_codes)
            return "", 0.0
        # For handwriting, paragraph=False (line-by-line) often works better first
        easy_results = reader.readtext(img, detail=1, paragraph=False)
        if not easy_results:
            easy_results = reader.readtext(img, detail=1, paragraph=True)
    except Exception as exc:
        logger.warning("EasyOCR readtext error: %s", exc, exc_info=True)
        return "", 0.0

    if not easy_results:
        return "", 0.0

    lines = []
    confidences = []
    for entry in easy_results:
        text = None
        confidence = 0.5

        if isinstance(entry, (list, tuple)):
            if len(entry) == 3:
                _, text, confidence = entry
            elif len(entry) >= 2:
                text = entry[1] if isinstance(entry[1], str) else entry[0]
                try:
                    confidence = float(entry[-1])
                except Exception:
                    confidence = 0.5
        elif isinstance(entry, str):
            text = entry

        if not isinstance(text, str):
            continue

        cleaned = text.strip()
        if cleaned:
            lines.append(cleaned)
            confidences.append(float(confidence))

    combined_text = "\n".join(lines).strip()
    if confidences:
        avg_confidence = sum(confidences) / len(confidences)
        avg_confidence = max(0.0, min(avg_confidence, 1.0))
    else:
        avg_confidence = 0.0
    return combined_text, avg_confidence


def _get_paddleocr_reader(language_codes=None):
    """Get or create PaddleOCR reader instance"""
    global _paddleocr_reader
    
    if not PADDLEOCR_AVAILABLE or PaddleOCR is None:
        return None
    
    if _paddleocr_reader is None:
        try:
            # Map language codes for PaddleOCR (uses 'en', 'ch', 'korean', etc.)
            lang = 'en'  # Default to English
            if language_codes:
                # Map Tesseract codes to PaddleOCR codes
                lang_map = {
                    'eng': 'en',
                    'hin': 'hi',  # Hindi
                    'ch': 'ch',   # Chinese
                    'korean': 'korean',
                }
                # Use first language code
                first_lang = language_codes[0] if language_codes else 'eng'
                lang = lang_map.get(first_lang, 'en')
            
            _paddleocr_reader = PaddleOCR(
                use_angle_cls=True,
                lang=lang,
                use_gpu=PADDLEOCR_USE_GPU,
                show_log=False
            )
        except Exception as e:
            print(f"PaddleOCR initialization error: {str(e)}")
            return None
    
    return _paddleocr_reader


def run_paddleocr(image_path, language_codes=None):
    """Run PaddleOCR on an image using OpenCV preprocessing"""
    if not PADDLEOCR_AVAILABLE or PaddleOCR is None:
        return "", 0.0
    
    try:
        reader = _get_paddleocr_reader(language_codes)
        if reader is None:
            return "", 0.0
        
        # Use OpenCV to read and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            return "", 0.0
        
        # PaddleOCR can work with numpy array directly
        # Format: [[[bbox], (text, confidence)], ...]
        results = reader.ocr(img, cls=True)
        
        if not results or not results[0]:
            return "", 0.0
        
        lines = []
        confidences = []
        
        for line in results[0]:
            if line and len(line) >= 2:
                # line[1] is (text, confidence)
                text = line[1][0] if isinstance(line[1], (list, tuple)) and len(line[1]) > 0 else ""
                confidence = line[1][1] if isinstance(line[1], (list, tuple)) and len(line[1]) > 1 else 0.5
                
                if text and isinstance(text, str):
                    cleaned = text.strip()
                    if cleaned:
                        lines.append(cleaned)
                        confidences.append(float(confidence))
        
        combined_text = "\n".join(lines).strip()
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            avg_confidence = max(0.0, min(avg_confidence, 1.0))
        else:
            avg_confidence = 0.0
        
        return combined_text, avg_confidence
        
    except Exception as exc:
        print(f"PaddleOCR error: {str(exc)}")
        return "", 0.0


# Medical dictionary for common prescription terms
MEDICATION_DICT = {
    # Common medications - Updated for Indian market
    "amoxicillin": ["amox", "amoxil", "amoxicil", "amoxicilin", "mox", "novamox", "almoxi", "wymox", "moxikind", "moxikind cv", "moxikind-cv"],
    "amoxicillin + clavulanic acid": ["moxikind cv", "moxikind-cv", "moxikindcv", "augmentin", "moxclav", "megaclox", "clavam", "hiclav", "clavum", "amoxiclav"],
    "paracetamol": ["paracet", "parcetamol", "acetaminophen", "tylenol", "crocin", "panadol", "dolo", "metacin", "calpol", "sumo", "febrex", "acepar", "pacimol", "fepanil"],
    "dextromethorphan": ["tuss dx", "tuss-dx", "tussdx", "tuss", "dx"],
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

COMMON_PRESCRIPTION_TERMS = {
    "tab",
    "tabs",
    "tablet",
    "tablets",
    "cap",
    "caps",
    "capsule",
    "capsules",
    "syr",
    "syrup",
    "spray",
    "nasal",
    "drops",
    "ointment",
    "cream",
    "gel",
    "suspension",
    "injection",
    "inj",
    "apply",
    "take",
    "before",
    "after",
    "meals",
    "meal",
    "daily",
    "night",
    "morning",
    "evening",
    "noon",
    "mg",
    "ml",
    "once",
    "twice",
    "thrice",
    "spr",
    "syp",
    "spondex",
    "spray",
    "xylometazoline",
    "nasal",
    "spray",
    "nasoclear",
    "syrup",
    "cv",
    "xr",
    "sr",
    "plus",
    "forte",
}

MEDICAL_SIGNAL_TERMS = set(term.lower() for term in COMMON_PRESCRIPTION_TERMS)
for med_name, aliases in MEDICATION_DICT.items():
    for token in re.findall(r'\b[a-z]+\b', med_name.lower()):
        MEDICAL_SIGNAL_TERMS.add(token)
    for alias in aliases:
        for token in re.findall(r'\b[a-z]+\b', alias.lower()):
            MEDICAL_SIGNAL_TERMS.add(token)

def apply_medical_dictionary_correction(text):
    """Apply medical dictionary correction to OCR text"""
    if not text:
        return text
        
    words = re.findall(r'\b\w+\b', text.lower())
    corrected_text = text
    
    for word in words:
        # Skip very short words or numbers
        if len(word) < 3 or word.isdigit():
            continue
            
        # Find the best match in our medication dictionary
        best_match = None
        best_score = 0
        best_key = None
        
        for key, aliases in MEDICATION_DICT.items():
            # Check the key itself
            score = fuzz.ratio(word, key.lower())
            if score > best_score and score > 75:  # Threshold of 75%
                best_score = score
                best_match = key
                best_key = key
                
            # Check aliases
            for alias in aliases:
                score = fuzz.ratio(word, alias.lower())
                if score > best_score and score > 75:  # Threshold of 75%
                    best_score = score
                    best_match = key  # Use the standardized term, not the alias
                    best_key = key
        
        if best_match and best_score > 75:
            # Replace the word with the correct spelling, maintaining original case
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            corrected_text = pattern.sub(best_match, corrected_text)
    
    return corrected_text

# ---------------------------------------------------------------------------
# Image preprocessing for better OCR accuracy
# ---------------------------------------------------------------------------
# Min width/height (pixels) below which we upscale to improve Tesseract accuracy
_MIN_EDGE_FOR_UPSCALE = 600
# Target scale: ensure shortest edge at least this for printed text
_TARGET_MIN_EDGE = 1200
# Max edge to avoid huge memory use
_MAX_EDGE = 3200


def _ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert to grayscale if needed."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _resize_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Resize image for better accuracy: upscale small images so Tesseract has
    enough resolution; cap max size to avoid OOM. Low-res images benefit most.
    """
    h, w = image.shape[:2]
    min_edge = min(h, w)
    max_edge = max(h, w)
    if min_edge >= _MIN_EDGE_FOR_UPSCALE and max_edge <= _MAX_EDGE:
        return image
    scale = 1.0
    if min_edge < _MIN_EDGE_FOR_UPSCALE and min_edge > 0:
        scale = max(scale, _TARGET_MIN_EDGE / min_edge)
    if max_edge > _MAX_EDGE and max_edge > 0:
        scale = min(scale, _MAX_EDGE / max_edge)
    if scale == 1.0:
        return image
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def preprocess_image_advanced(
    image_path: str,
    *,
    denoise_strength: int = 10,
    use_bilateral: bool = True,
    blur_kernel: int = 1,
    adaptive_block: int = 11,
    adaptive_c: int = 2,
    use_otsu_fallback: bool = True,
    morph_kernel_size: int = 2,
    save_dir: str | None = None,
) -> dict | None:
    """
    Production-ready image preprocessing for OCR.

    1. Load image and convert to grayscale (reduces noise, standard for Tesseract).
    2. Resize: upscale small images for better accuracy; cap size for performance.
    3. Noise removal: fastNlMeansDenoising (strong on text) + optional bilateral
       (preserves edges while smoothing).
    4. Optional light Gaussian blur (kernel 1 = none) to reduce salt-and-pepper.
    5. Binarization: adaptive threshold (good for uneven lighting); Otsu fallback
       for high contrast images.
    6. Morphology: small open (remove thin noise) then optional dilate to close
       character gaps.

    Returns dict with keys: original, enhanced (path), processed_image (array),
    and optionally enhanced_inverted (path) for inverted variant. Returns None
    on error.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.error("Failed to load image: %s", image_path)
            return None

        gray = _ensure_grayscale(image)
        gray = _resize_for_ocr(gray)
        logger.debug("Image shape after resize: %s", gray.shape)

        # Noise removal: fastNlMeansDenoising is very effective for text
        denoised = cv2.fastNlMeansDenoising(
            gray, None, h=denoise_strength, templateWindowSize=7, searchWindowSize=21
        )
        if use_bilateral:
            denoised = cv2.bilateralFilter(denoised, 9, 75, 75)

        # Light blur only if kernel > 1 (reduces noise, can blur text if too strong)
        if blur_kernel > 1 and blur_kernel % 2 == 1:
            denoised = cv2.GaussianBlur(denoised, (blur_kernel, blur_kernel), 0)

        # Adaptive thresholding: better for prescriptions with uneven lighting
        if adaptive_block % 2 != 1:
            adaptive_block += 1
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, adaptive_block, adaptive_c
        )
        # Otsu fallback: use when image has bimodal intensity (e.g. clear text on white)
        if use_otsu_fallback:
            otsu_val, otsu_thresh = cv2.threshold(
                denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            # Prefer adaptive; optionally blend or choose by variance
            if np.std(otsu_thresh) > np.std(thresh):
                thresh = otsu_thresh

        # Morphology: open (erode then dilate) removes small noise; dilate closes gaps
        k = max(1, morph_kernel_size)
        kernel = np.ones((k, k), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        processed = cv2.dilate(processed, np.ones((2, 2), np.uint8), iterations=1)

        base_name = os.path.basename(image_path)
        dir_name = save_dir or os.path.dirname(image_path)
        enhanced_name = f"enhanced_{base_name}"
        enhanced_path = os.path.join(dir_name, enhanced_name)
        cv2.imwrite(enhanced_path, processed)
        logger.debug("Saved enhanced image: %s", enhanced_path)

        result = {
            "original": image_path,
            "enhanced": enhanced_path,
            "processed_image": processed,
        }

        # Inverted variant (white text on black) helps in some cases
        inverted = cv2.bitwise_not(processed)
        inv_path = os.path.join(dir_name, f"enhanced_inv_{base_name}")
        cv2.imwrite(inv_path, inverted)
        result["enhanced_inverted"] = inv_path

        return result
    except Exception as e:
        logger.exception("Image preprocessing failed for %s: %s", image_path, e)
        return None


def preprocess_image(image_path):
    """
    Simple and effective image preprocessing for prescription OCR.
    Uses the advanced pipeline (grayscale, denoise, adaptive threshold, resize, morphology).
    Returns dict with original, enhanced, processed_image, and optionally enhanced_inverted; or None on error.
    """
    try:
        result = preprocess_image_advanced(image_path)
        if result is None:
            return None
        return {
            "original": result["original"],
            "enhanced": result["enhanced"],
            "processed_image": result["processed_image"],
            "enhanced_inverted": result.get("enhanced_inverted"),
        }
    except Exception as e:
        logger.exception("preprocess_image failed: %s", e)
        return None

def run_multiple_ocr_passes(image_data, language_codes=None):
    """
    Run multiple OCR passes with different preprocessing variants and Tesseract configs.
    Uses enhanced, original, and optionally enhanced_inverted; each with multiple
    --oem/--psm configs. Passes language_codes through to Tesseract for multi-language.
    """
    try:
        configs = build_tesseract_configs(language_codes)
        results = []
        seen_variants = set()

        # Prefer enhanced (preprocessed), then original; include inverted if available
        ocr_targets = [
            (image_data.get("enhanced"), 1.0),
            (image_data.get("original"), 0.9),
        ]
        if image_data.get("enhanced_inverted"):
            ocr_targets.append((image_data["enhanced_inverted"], 0.85))

        for image_path, path_weight in ocr_targets:
            if not image_path or not os.path.exists(image_path):
                continue
            for config, config_weight in configs:
                result_text = extract_text_tesseract(
                    image_path, config=config, language_codes=language_codes
                )
                normalized = re.sub(r"\s+", " ", result_text.strip().lower()) if result_text else ""
                if normalized and normalized not in seen_variants:
                    seen_variants.add(normalized)
                    results.append((result_text.strip(), path_weight * config_weight))

        # Legacy fallback: create inverted from enhanced if no precomputed inverted
        if not results and image_data.get("enhanced"):
            enhanced_img = cv2.imread(image_data["enhanced"])
            if enhanced_img is not None:
                inverted = cv2.bitwise_not(enhanced_img)
                dir_name = os.path.dirname(image_data["enhanced"])
                inverted_path = os.path.join(dir_name, "inverted_temp.jpg")
                cv2.imwrite(inverted_path, inverted)
                try:
                    for config, config_weight in configs:
                        result_text = extract_text_tesseract(
                            inverted_path, config=config, language_codes=language_codes
                        )
                        normalized = re.sub(r"\s+", " ", result_text.strip().lower()) if result_text else ""
                        if normalized and normalized not in seen_variants:
                            seen_variants.add(normalized)
                            results.append((result_text.strip(), 0.85 * config_weight))
                finally:
                    try:
                        os.remove(inverted_path)
                    except Exception:
                        pass

        if not results:
            logger.debug("No text extracted in any OCR pass")
            return [("", 0.0)]
        return results
    except Exception as e:
        logger.exception("Error in OCR passes: %s", e)
        return [("", 0.0)]

def score_text_quality(text):
    """Heuristic score to estimate OCR text quality"""
    if not text:
        return 0.0

    collapsed = re.sub(r'\s+', ' ', text.strip())
    if not collapsed:
        return 0.0

    alnum_chars = [c for c in collapsed if c.isalnum()]
    alpha_ratio = (sum(c.isalpha() for c in alnum_chars) / len(collapsed)) if collapsed else 0.0

    words = re.findall(r'\b\w+\b', collapsed)
    if not words:
        return alpha_ratio

    unique_ratio = len(set(w.lower() for w in words)) / len(words)
    length_bonus = min(len(words), 60) / 60.0

    return (alpha_ratio * 0.5) + (unique_ratio * 0.3) + (length_bonus * 0.2)


def estimate_medical_signal(text):
    if not text:
        return 0.0
    tokens = re.findall(r'\b[a-z]+\b', text.lower())
    if not tokens:
        return 0.0
    hits = sum(1 for token in tokens if token in MEDICAL_SIGNAL_TERMS)
    return hits / len(tokens)


def evaluate_text_candidate(text):
    quality = score_text_quality(text)
    medical_signal = estimate_medical_signal(text)
    combined = (quality * 0.7) + (medical_signal * 0.3)
    return combined, quality, medical_signal


def _coerce_setting(value, default):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if numeric < 0.0:
        return 0.0
    if numeric > 1.0:
        return 1.0
    return numeric


def combine_ocr_results(results):
    """Select the highest-quality OCR result and return aggregated confidence"""
    try:
        if not results:
            return "", 0.0

        best_text = ""
        best_score = -1.0
        confidence_scores = []

        for text, confidence in results:
            cleaned_text = text.strip() if text else ""
            if not cleaned_text:
                continue

            confidence_scores.append(float(confidence))
            quality_score = score_text_quality(cleaned_text)

            if quality_score > best_score:
                best_score = quality_score
                best_text = cleaned_text

        if not best_text:
            return "", 0.0

        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

        return best_text, avg_confidence

    except Exception as e:
        print(f"Error combining OCR results: {str(e)}")
        return "", 0.0

def extract_medical_entities(text):
    """Extract medical entities from the text"""
    medications = []
    dosages = []
    frequencies = []
    routes = []
    
    # Use spaCy for entity recognition if available
    if nlp:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ["CHEMICAL", "DRUG", "MEDICATION"]:
                # Extract only the medication name without dosage or frequency
                med_name = re.sub(r'\s+\d+\s*\w*\b', '', ent.text) # Remove numbers and units
                med_name = re.sub(r'\b(once|twice|three|four)(\s+times)?\s+(daily|a\s+day)\b', '', med_name, flags=re.IGNORECASE)
                med_name = re.sub(r'\b(every|each)\s+(morning|evening|night|day|hour|hourly)\b', '', med_name, flags=re.IGNORECASE)
                med_name = re.sub(r'\b(qd|bid|tid|qid|prn|od|q\d+h)\b', '', med_name, flags=re.IGNORECASE)
                med_name = med_name.strip()
                if med_name and len(med_name) > 2:  # Ensure we have a reasonable name (not just a unit or directive)
                    medications.append(med_name)
    
    # Extract medications using our dictionary
    for key in MEDICATION_DICT.keys():
        # Skip dosage units, frequency terms, routes, and instructions
        if key in ["milligram", "microgram", "gram", "milliliter", 
                   "once daily", "twice daily", "three times daily", "four times daily",
                   "every morning", "every night", "every hour", "every 4 hours",
                   "every 6 hours", "every 8 hours", "every 12 hours", "as needed",
                   "by mouth", "intravenous", "intramuscular", "subcutaneous",
                   "sublingual", "topical", "inhalation",
                   "with food", "before meals", "after meals", "with water",
                   "do not crush", "take with plenty of water", "dissolve in water",
                   "until finished", "shake well"]:
            continue
            
        if re.search(r'\b' + re.escape(key) + r'\b', text, re.IGNORECASE):
            medications.append(key)
        else:
            # Check aliases, but only for medication items
            for alias in MEDICATION_DICT[key]:
                if re.search(r'\b' + re.escape(alias) + r'\b', text, re.IGNORECASE):
                    medications.append(key)  # Add the standardized term
                    break
    
    # Additional pattern-based extraction for medications
    # Look for common medication patterns in the text
    medication_patterns = [
        r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+(?:tablet|capsule|syrup|injection|tab|cap|syp|inj)\b',
        r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)(?:\s+\d+(?:mg|ml|g|mcg|units?)?)\b',
        r'\b(?:tab|tablet|cap|capsule|syp|syrup)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b',
        r'\b([A-Z][a-zA-Z]+\d+)\b',  # Medicine names with numbers (e.g., "Paracetamol500")
        r'\b([A-Z]{2,}[0-9]{2,})\b',  # Common medicine formats (e.g., "CIP500", "DOLO650")
        r'\b([A-Z]{2,}-[A-Z0-9]+)\b',  # Medicine names with hyphens (e.g., "MED-X")
        r'\b([A-Z][a-zA-Z]{3,}(?:\s+[A-Z][a-zA-Z]{2,})*)\s+\d+\s*(?:mg|ml|g)\b',  # Medicine name followed by dosage
        r'\b(Tab|Cap|Syp|Inj)\.?\s+([A-Z][a-zA-Z]{3,})\b',  # Indian prescription format (e.g., "Tab. Paracip")
        r'\b([A-Z][a-zA-Z]{4,})\s+(?:\d+(?:mg|ml|g)?)\b',  # Medicine name followed by dosage
        r'\b([A-Z][a-zA-Z]{3,})\s+(\d+mg|\d+ml|\d+g)\b',  # Medicine names with dosage after (e.g., "Aten 50mg")
        r'\b([A-Z][a-zA-Z]+\d{3,})\b',  # Medicine names with strength numbers (e.g., "Dolo650")
        r'\b([A-Z][a-zA-Z]{4,})\s+(\d{1,3}(?:\.\d{1,2})?\s*(?:mg|ml|g))\b',  # Common Indian medicine formats (e.g., "Rabeprazole 20mg")
        r'\b([A-Z]{5,}[0-9]*)\b',  # Standalone capitalized medicine names (e.g., "ATENOLOL", "CIPROFLOXACIN")
    ]
    
    for pattern in medication_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            med_name = match.group(1) if len(match.groups()) >= 1 else match.group(0)
            # If it's a tuple with two groups (like "Tab Paracip"), take the medicine name part
            if len(match.groups()) >= 2:
                med_name = match.group(2)  # Take the second group which is the medicine name
            # Basic cleaning
            med_name = re.sub(r'\s+', ' ', med_name.strip())
            if len(med_name) > 2 and med_name.lower() not in ['the', 'and', 'for', 'with', 'take', 'have', 'not', 'this', 'that', 'will', 'should', 'would', 'could', 'may', 'might', 'must', 'can', 'does', 'did', 'do', 'is', 'are', 'was', 'were', 'been', 'being', 'be', 'has', 'had', 'having', 'get', 'got', 'getting', 'make', 'made', 'making', 'go', 'went', 'going', 'come', 'came', 'coming', 'see', 'saw', 'seeing', 'know', 'knew', 'knowing', 'think', 'thought', 'thinking', 'say', 'said', 'saying', 'tell', 'told', 'telling', 'ask', 'asked', 'asking', 'give', 'gave', 'giving', 'put', 'turn', 'turned', 'turning', 'keep', 'kept', 'keeping', 'let', 'lets', 'letting', 'begin', 'began', 'beginning', 'start', 'started', 'starting', 'continue', 'continued', 'continuing', 'try', 'tried', 'trying', 'need', 'needed', 'needing', 'want', 'wanted', 'wanting', 'like', 'liked', 'liking', 'seem', 'seemed', 'seeming', 'become', 'became', 'becoming', 'leave', 'left', 'leaving', 'feel', 'felt', 'feeling', 'appear', 'appeared', 'appearing', 'look', 'looked', 'looking', 'hear', 'heard', 'hearing', 'play', 'played', 'playing', 'run', 'ran', 'running', 'move', 'moved', 'moving', 'live', 'lived', 'living', 'believe', 'believed', 'believing', 'hold', 'held', 'holding', 'bring', 'brought', 'bringing', 'happen', 'happened', 'happening', 'write', 'wrote', 'writing', 'sit', 'sat', 'sitting', 'stand', 'stood', 'standing', 'lose', 'lost', 'losing', 'pay', 'paid', 'paying', 'meet', 'met', 'meeting', 'include', 'included', 'including', 'set', 'setting', 'learn', 'learned', 'learning', 'change', 'changed', 'changing', 'lead', 'led', 'leading', 'understand', 'understood', 'understanding', 'watch', 'watched', 'watching', 'follow', 'followed', 'following', 'stop', 'stopped', 'stopping', 'create', 'created', 'creating', 'speak', 'spoke', 'speaking', 'read', 'spend', 'spent', 'spending', 'grow', 'grew', 'growing', 'open', 'opened', 'opening', 'walk', 'walked', 'walking', 'win', 'won', 'winning', 'teach', 'taught', 'teaching', 'offer', 'offered', 'offering', 'remember', 'remembered', 'remembering', 'consider', 'considered', 'considering', 'buy', 'bought', 'buying', 'serve', 'served', 'serving', 'die', 'died', 'dying', 'send', 'sent', 'sending', 'build', 'built', 'building', 'stay', 'stayed', 'staying', 'fall', 'fell', 'falling', 'cut', 'rise', 'rose', 'rising', 'drive', 'drove', 'driving', 'break', 'broke', 'breaking', 'choose', 'chose', 'choosing', 'forget', 'forgot', 'forgetting', 'drink', 'drank', 'drinking', 'eat', 'ate', 'eating', 'find', 'found', 'finding', 'fly', 'flew', 'flying', 'hide', 'hid', 'hiding', 'hit', 'lay', 'laid', 'laying', 'lie', 'ring', 'rang', 'ringing', 'shake', 'shook', 'shaking', 'sing', 'sang', 'singing', 'sink', 'sank', 'sinking', 'stick', 'stuck', 'sticking', 'strike', 'struck', 'striking', 'tear', 'tore', 'tearing', 'throw', 'threw', 'throwing', 'wake', 'woke', 'waking', 'wear', 'wore', 'wearing', 'diagnosis', 'symptom', 'symptoms', 'treatment', 'therapy', 'procedure', 'operation', 'surgery', 'test', 'exam', 'checkup', 'consultation', 'appointment', 'followup', 'review', 'monitoring', 'monitor', 'check', 'screening', 'screen', 'evaluation', 'assessment', 'prescribe', 'prescribing', 'prescriber', 'pharmacist', 'pharmacy', 'pharmaceutical', 'pharmaceuticals', 'pharma', 'medicinal', 'medicinals', 'therapeutic', 'therapeutics', 'clinical', 'clinically', 'medical', 'medically', 'health', 'healthcare', 'care', 'hospital', 'clinic', 'doctor', 'physician', 'specialist', 'nurse', 'nursing', 'patient', 'patients', 'client', 'clients', 'person', 'people', 'human', 'humans', 'individual', 'individuals', 'subject', 'subjects']:
                medications.append(med_name.title())
    
    # Pattern for dosage (number + unit)
    dosage_pattern = r'\b(\d+[\.\d]*)\s*(mg|mcg|mL|g|mg/mL|mEq|units|tablets?|caps?|syps?|injs?)\b'
    dosage_matches = re.finditer(dosage_pattern, text, re.IGNORECASE)
    for match in dosage_matches:
        dosages.append(match.group(0))
    
    # Pattern for frequencies
    freq_patterns = [
        r'\b(once|twice|three times|four times)\s+daily\b',
        r'\b(q\.?d|b\.?i\.?d|t\.?i\.?d|q\.?i\.?d)\b',
        r'\b(every|each)\s+(\d+)\s+(hours?|days?)\b',
        r'\b(q)(\d+)(h)\b',
        r'\b(\d+-\d+-\d+)\b',  # Indian prescription format like 1-0-1
        r'\bprn\b',
        r'\bas needed\b',
        r'\b(od|bd|tds|qid)\b',  # Common medical abbreviations for frequency
    ]
    
    for pattern in freq_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            frequencies.append(match.group(0))
    
    # Pattern for routes of administration
    route_patterns = [
        r'\b(oral(ly)?|by mouth|p\.?o\.)\b',
        r'\b(intravenous|i\.?v\.)\b',
        r'\b(intramuscular|i\.?m\.)\b',
        r'\b(subcutaneous|s\.?c\.|sub-q)\b',
        r'\b(topical(ly)?)\b',
        r'\b(sublingual|s\.?l\.)\b',
        r'\b(tablet|capsule|syrup|injection)\b',
        r'\b(tab|cap|syp|inj)\b',  # Common abbreviations
    ]
    
    for pattern in route_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            routes.append(match.group(0))
    
    # Clean medication names to remove any residual dosage or frequency info
    clean_medications = []
    for med in medications:
        # Remove dosage and frequency information
        clean_med = re.sub(r'\s+\d+\s*\w*\b', '', med).strip()
        clean_med = re.sub(r'\b(once|twice|three|four)(\s+times)?\s+(daily|a\s+day)\b', '', clean_med, flags=re.IGNORECASE).strip()
        clean_med = re.sub(r'\b(every|each)\s+(morning|evening|night|day|hour|hourly)\b', '', clean_med, flags=re.IGNORECASE).strip()
        clean_med = re.sub(r'\b(qd|bid|tid|qid|prn|od|bd|tds|qid|q\d+h)\b', '', clean_med, flags=re.IGNORECASE).strip()
        
        if clean_med and len(clean_med) > 2:  # Ensure we have a meaningful name
            clean_medications.append(clean_med)
    
    # If no medications found but text suggests medications, add a general entry
    if not clean_medications and re.search(r'\b(?:medication|prescription|rx|tablet|capsule|syrup|take|dose|mg|ml|g)\b', text, re.IGNORECASE):
        clean_medications = ['Medication Not Clearly Identified']
        if not dosages:
            dosages = ['See prescription text']
        if not frequencies:
            frequencies = ['As prescribed']
    
    return {
        "medications": list(set(clean_medications)),
        "dosages": list(set(dosages)),
        "frequencies": list(set(frequencies)),
        "routes": list(set(routes))
    }

def extract_medicines_strict(ocr_text):
    """
    STRICT medicine extraction following safety rules:
    - Only extract medicines explicitly or approximately present in OCR text
    - Do NOT guess or insert common drugs
    - Include evidence field with OCR word that triggered detection
    - Use "Not mentioned" for missing fields
    - Use "Unclear (OCR: <word>)" for unclear medicine names
    - Do NOT fabricate dosage, frequency, or duration
    - Return empty array if no valid medicine tokens exist
    
    Returns: JSON structure with medicines array
    """
    if not ocr_text or not isinstance(ocr_text, str):
        return {"medicines": []}
    
    text_lower = ocr_text.lower()
    medicines = []
    
    # Medicine indicators that suggest a medicine might be present
    medicine_indicators = [
        r'\b(tab|tabs|tablet|tablets)\b',
        r'\b(cap|caps|capsule|capsules)\b',
        r'\b(syp|syrup|syrups|sup)\b',  # Include "sup" as OCR error for "syp"
        r'\b(inj|injection|injections)\b',
        r'\b(rx|prescription)\b',
        r'\b(mg|ml|g|mcg|units?)\b',  # Dosage units that often indicate medicines
        r'\b(take|dose|dosed)\b',  # Action words that suggest medicines
        r'\b(\d+-\d+-\d+)\b',  # Indian prescription format like 1-0-1
    ]
    
    # Check if any medicine indicators exist
    has_indicators = any(re.search(pattern, text_lower) for pattern in medicine_indicators)
    
    # Extract potential medicine names from text
    # Look for words that appear after medicine indicators or standalone drug-like words
    words = re.findall(r'\b[a-z]{3,}\b', text_lower)
    
    # If no indicators and no substantial words, but text suggests medications, add a general entry
    if not has_indicators and len(words) < 3:
        # Check if text suggests medications are present
        if re.search(r'\b(?:medication|prescription|rx|tablet|capsule|syrup|take|dose|mg|ml|g)\b', text_lower):
            return {
                "medicines": [{
                    "name": "Medication Not Clearly Identified",
                    "strength": "See prescription text",
                    "frequency": "As prescribed",
                    "duration": "Not mentioned",
                    "route": "Not mentioned",
                    "evidence": "Text suggests presence of medications"
                }]
            }
        return {"medicines": []}
    
    # Pattern to find medicine entries: indicator followed by potential medicine name
    # Format: Tab/Cap/Syp/Inj [medicine_name] [dosage] [frequency]
    medicine_patterns = [
        # Tab/Cap/Syp/Inj followed by medicine name (most reliable) - improved to capture multi-word names
        r'\b(tab|tabs|tablet|tablets|cap|caps|capsule|capsules|syp|syrup|syrups|sup|inj|injection|injections)\s+([a-z]{3,}(?:\s+[a-z]{1,4})?(?:\s+[a-z]{2,})?)\s*',
        # Medicine name followed by dosage (indicates medicine) - improved to handle numbers without units
        r'\b([a-z]{4,}(?:\s+[a-z]{1,4})?)\s+(\d{3,}|\d+[\.\d]*\s*(?:mg|mcg|ml|g|mg/ml|meq|units?))\b',
        # Medicine name with CV, DX, etc. patterns
        r'\b([a-z]{4,})\s+(cv|dx|sr|er|xr|plus|forte)\s*(\d+)?\b',
        # Generic pattern for capitalized medicine names
        r'\b([A-Z][a-z]{3,}(?:\s+[A-Z][a-z]{2,})*)\b',
        # Pattern for medicine names with numbers (e.g., "Medicine123")
        r'\b([A-Z][a-z]+\d+)\b',
        # Pattern for common medicine formats with numbers (e.g., "CIP500", "DOLO650")
        r'\b([A-Z]{2,}[0-9]{2,})\b',
        # Pattern for medicine names with hyphens (e.g., "MED-X")
        r'\b([A-Z]{2,}-[A-Z0-9]+)\b',
        # Pattern for Indian prescription format (e.g., "Tab Paracip")
        r'\b(tab|cap|syp|inj)\s+([A-Z][a-z]+)\b',
        # Pattern for common medicine abbreviations (e.g., "Tab Crocin")
        r'\b(tab|cap|syp|inj)\s+([A-Z][a-z]{3,})\b',
        # Pattern for medicine names with dosage after (e.g., "Aten 50mg")
        r'\b([A-Z][a-z]{3,})\s+(\d+mg|\d+ml|\d+g)\b',
        # Pattern for medicine names with strength numbers (e.g., "Dolo650")
        r'\b([A-Z][a-z]+\d{3,})\b',
        # Pattern for common Indian medicine formats (e.g., "Rabeprazole 20mg")
        r'\b([A-Z][a-z]{4,})\s+(\d{1,3}(?:\.\d{1,2})?\s*(?:mg|ml|g))\b',
    ]
    
    found_medicines = {}
    
    # Extract medicine candidates with context
    for pattern in medicine_patterns:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            if len(match.groups()) >= 1:
                # Get the medicine name candidate - handle multi-word names better
                if len(match.groups()) >= 2:
                    # Pattern like: tab moxikind cv or moxikind cv 625
                    med_candidate = match.group(1)  # First group is usually the name
                    if len(match.groups()) >= 3:
                        # Handle cases like "moxikind cv 625" - combine name parts
                        med_candidate = f"{match.group(1)} {match.group(2)}".strip()
                else:
                    med_candidate = match.group(1)
                
                # Clean up the candidate - remove trailing numbers that are dosages
                med_candidate = re.sub(r'\s+\d{3,}$', '', med_candidate).strip()
                
                # Skip common non-medicine words
                skip_words = {
                    'tablet', 'tablets', 'capsule', 'capsules', 'syrup', 'syrups',
                    'injection', 'injections', 'prescription', 'doctor', 'patient',
                    'morning', 'evening', 'night', 'daily', 'before', 'after',
                    'meals', 'food', 'water', 'times', 'days', 'weeks', 'months',
                    'take', 'apply', 'use', 'with', 'without', 'every', 'each',
                    'the', 'and', 'for', 'this', 'that', 'have', 'not', 'will', 'can',
                    'should', 'would', 'could', 'may', 'might', 'must', 'does', 'did',
                    'do', 'is', 'are', 'was', 'were', 'been', 'being', 'be', 'has', 'had',
                    'having', 'get', 'got', 'getting', 'make', 'made', 'making', 'go', 'went',
                    'going', 'come', 'came', 'coming', 'see', 'saw', 'seeing', 'know', 'knew',
                    'knowing', 'think', 'thought', 'thinking', 'say', 'said', 'saying', 'tell',
                    'told', 'telling', 'ask', 'asked', 'asking', 'give', 'gave', 'giving',
                    'put', 'turn', 'turned', 'turning', 'keep', 'kept', 'keeping', 'let', 'lets',
                    'letting', 'begin', 'began', 'beginning', 'start', 'started', 'starting',
                    'continue', 'continued', 'continuing', 'try', 'tried', 'trying', 'need',
                    'needed', 'needing', 'want', 'wanted', 'wanting', 'like', 'liked', 'liking',
                    'seem', 'seemed', 'seeming', 'become', 'became', 'becoming', 'leave', 'left',
                    'leaving', 'feel', 'felt', 'feeling', 'appear', 'appeared', 'appearing',
                    'look', 'looked', 'looking', 'hear', 'heard', 'hearing', 'play', 'played',
                    'playing', 'run', 'ran', 'running', 'move', 'moved', 'moving', 'live', 'lived',
                    'living', 'believe', 'believed', 'believing', 'hold', 'held', 'holding',
                    'bring', 'brought', 'bringing', 'happen', 'happened', 'happening', 'write',
                    'wrote', 'writing', 'sit', 'sat', 'sitting', 'stand', 'stood', 'standing',
                    'lose', 'lost', 'losing', 'pay', 'paid', 'paying', 'meet', 'met', 'meeting',
                    'include', 'included', 'including', 'set', 'setting', 'learn', 'learned',
                    'learning', 'change', 'changed', 'changing', 'lead', 'led', 'leading',
                    'understand', 'understood', 'understanding', 'watch', 'watched', 'watching',
                    'follow', 'followed', 'following', 'stop', 'stopped', 'stopping', 'create',
                    'created', 'creating', 'speak', 'spoke', 'speaking', 'read', 'spend', 'spent',
                    'spending', 'grow', 'grew', 'growing', 'open', 'opened', 'opening', 'walk',
                    'walked', 'walking', 'win', 'won', 'winning', 'teach', 'taught', 'teaching',
                    'offer', 'offered', 'offering', 'remember', 'remembered', 'remembering',
                    'consider', 'considered', 'considering', 'buy', 'bought', 'buying', 'serve',
                    'served', 'serving', 'die', 'died', 'dying', 'send', 'sent', 'sending', 'build',
                    'built', 'building', 'stay', 'stayed', 'staying', 'fall', 'fell', 'falling',
                    'cut', 'rise', 'rose', 'rising', 'drive', 'drove', 'driving', 'break', 'broke',
                    'breaking', 'choose', 'chose', 'choosing', 'forget', 'forgot', 'forgetting',
                    'drink', 'drank', 'drinking', 'eat', 'ate', 'eating', 'find', 'found', 'finding',
                    'fly', 'flew', 'flying', 'hide', 'hid', 'hiding', 'hit', 'lay', 'laid', 'laying',
                    'lie', 'ring', 'rang', 'ringing', 'shake', 'shook', 'shaking', 'sing', 'sang',
                    'singing', 'sink', 'sank', 'sinking', 'stick', 'stuck', 'sticking', 'strike',
                    'struck', 'striking', 'tear', 'tore', 'tearing', 'throw', 'threw', 'throwing',
                    'wake', 'woke', 'waking', 'wear', 'wore', 'wearing',
                    # Additional common words to skip
                    'prescribed', 'medicine', 'medication', 'drug', 'treatment', 'therapy',
                    'dose', 'dosed', 'dosage', 'strength', 'frequency', 'duration',
                    'route', 'administration', 'instruction', 'direction', 'directions',
                    'as', 'per', 'when', 'how', 'why', 'what', 'where', 'who', 'which',
                    'but', 'or', 'if', 'then', 'else', 'than', 'so', 'too', 'very',
                    'just', 'only', 'also', 'even', 'still', 'yet', 'already', 'since',
                    'until', 'while', 'during', 'before', 'after', 'through', 'between',
                    'among', 'above', 'below', 'under', 'over', 'again', 'further',
                    'once', 'twice', 'thrice', 'first', 'second', 'third', 'last',
                    'next', 'previous', 'former', 'latter', 'same', 'different',
                    'other', 'another', 'such', 'same', 'similar', 'different',
                    'many', 'much', 'more', 'most', 'few', 'less', 'least',
                    'some', 'any', 'no', 'none', 'all', 'every', 'each',
                    'both', 'either', 'neither', 'one', 'two', 'three', 'four',
                    'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven',
                    'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
                    'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty',
                    'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety',
                    'hundred', 'thousand', 'million', 'billion', 'trillion',
                    # Common medical terms that are not medicines
                    'diagnosis', 'symptom', 'symptoms', 'treatment', 'therapy',
                    'procedure', 'operation', 'surgery', 'test', 'exam', 'checkup',
                    'consultation', 'appointment', 'followup', 'review', 'monitoring',
                    'monitor', 'check', 'screening', 'screen', 'evaluation', 'assessment',
                    'prescribe', 'prescribing', 'prescriber', 'pharmacist', 'pharmacy',
                    'pharmaceutical', 'pharmaceuticals', 'pharma', 'medicinal', 'medicinals',
                    'therapeutic', 'therapeutics', 'clinical', 'clinically', 'medical',
                    'medically', 'health', 'healthcare', 'care', 'hospital', 'clinic',
                    'doctor', 'physician', 'specialist', 'nurse', 'nursing', 'nurse',
                    'patient', 'patients', 'client', 'clients', 'person', 'people',
                    'human', 'humans', 'individual', 'individuals', 'subject', 'subjects'
                }
                
                if med_candidate.lower() in skip_words or len(med_candidate) < 3:
                    continue
                
                # Handle OCR errors: "c2" might be "cv", "sup" might be "syp"
                med_candidate_clean = med_candidate
                med_candidate_clean = re.sub(r'\bc2\b', 'cv', med_candidate_clean)
                med_candidate_clean = re.sub(r'\bsup\b', 'syp', med_candidate_clean)
                
                # Check if this looks like a medicine name (not a common word)
                # Use fuzzy matching to check against medication dictionary ONLY if word is similar
                evidence_word = med_candidate  # Keep original for evidence
                medicine_name = None
                confidence = "clear"
                
                # Use cleaned version for matching
                search_candidate = med_candidate_clean.lower()
                
                # Check against medication dictionary with fuzzy matching (75% threshold)
                best_match = None
                best_score = 0
                
                for key, aliases in MEDICATION_DICT.items():
                    # Skip non-medication entries
                    if key in ["milligram", "microgram", "gram", "milliliter",
                              "once daily", "twice daily", "three times daily", "four times daily",
                              "every morning", "every night", "every hour", "every 4 hours",
                              "every 6 hours", "every 8 hours", "every 12 hours", "as needed",
                              "by mouth", "intravenous", "intramuscular", "subcutaneous",
                              "sublingual", "topical", "inhalation",
                              "with food", "before meals", "after meals", "with water",
                              "do not crush", "take with plenty of water", "dissolve in water",
                              "until finished", "shake well"]:
                        continue
                    
                    # Check exact match or high similarity with cleaned candidate
                    if search_candidate == key.lower() or search_candidate.startswith(key.lower()) or key.lower().startswith(search_candidate):
                        best_match = key
                        best_score = 100
                        break
                    
                    # Check if cleaned candidate contains key or vice versa (for multi-word names)
                    if search_candidate in key.lower() or key.lower() in search_candidate:
                        if len(search_candidate) >= 4:  # Only if substantial match
                            best_match = key
                            best_score = 95
                            break
                    
                    # Check aliases
                    for alias in aliases:
                        alias_lower = alias.lower()
                        if search_candidate == alias_lower or search_candidate.startswith(alias_lower) or alias_lower.startswith(search_candidate):
                            best_match = key
                            best_score = 100
                            break
                        score = fuzz.ratio(search_candidate, alias_lower)
                        if score > best_score and score >= 70:  # Lowered threshold to 70 for better matching
                            best_score = score
                            best_match = key
                
                if best_match and best_score >= 70:
                    medicine_name = best_match
                    if best_score < 85:
                        confidence = "unclear"
                else:
                    # If no match found but word looks medicine-like, use the cleaned candidate
                    if len(med_candidate_clean) >= 4:
                        # Capitalize first letter of each word for better display
                        medicine_name = ' '.join(word.capitalize() for word in med_candidate_clean.split())
                        confidence = "unclear"
                    else:
                        continue  # Skip if doesn't look like medicine
                
                # Extract dosage near this medicine
                strength = "Not mentioned"
                # Look for dosage pattern near the medicine (within 100 chars for better context)
                match_start = match.start()
                context = text_lower[max(0, match_start-50):match_start+150]
                
                # Improved dosage patterns - handle numbers with/without units, and patterns like 5-5-5ml
                dosage_patterns = [
                    r'(\d+[\.\d]*)\s*(mg|mcg|ml|g|mg/ml|meq|units?)\b',  # Standard: 500mg, 5ml
                    r'(\d+-\d+-\d+)\s*(mg|mcg|ml|g)\b',  # Pattern: 5-5-5ml (three times)
                    r'\b(\d{3,})\b',  # Large numbers like 625, 500 (likely dosages)
                ]
                
                for dosage_pattern in dosage_patterns:
                    dosage_match = re.search(dosage_pattern, context)
                    if dosage_match:
                        if len(dosage_match.groups()) > 1 and dosage_match.group(2):
                            strength = f"{dosage_match.group(1)} {dosage_match.group(2)}"
                        elif dosage_match.group(1) and len(dosage_match.group(1)) >= 3:
                            # Large number without unit - likely a dosage
                            num = dosage_match.group(1)
                            if '-' in num:
                                # Pattern like 5-5-5ml - extract the number and unit if present
                                strength = num
                            else:
                                strength = num  # Will be marked as dosage number
                        break
                
                # Extract frequency near this medicine
                frequency = "Not mentioned"
                freq_patterns = [
                    r'\b(1-0-1|1-1-1|0-0-1|0-1-0)\b',  # Common Indian prescription format
                    r'\b(\d+-\d+-\d+)\b',  # Pattern like 5-5-5 (three times daily)
                    r'\b(od|qd|once\s+daily|once\s+a\s+day)\b',
                    r'\b(bd|bid|twice\s+daily|twice\s+a\s+day|2\s+times)\b',
                    r'\b(tds|tid|three\s+times\s+daily|3\s+times)\b',
                    r'\b(qid|qds|four\s+times\s+daily|4\s+times)\b',
                    r'\b(every\s+\d+\s+hours?|q\d+h)\b',
                    r'\b(prn|as\s+needed|sos)\b',
                    r'\b(^|\s)1(\s|$)\b',  # Standalone "1" often means once daily
                ]
                
                for freq_pattern in freq_patterns:
                    freq_match = re.search(freq_pattern, context, re.IGNORECASE)
                    if freq_match:
                        freq_text = freq_match.group(0).strip()
                        # Convert patterns to readable format
                        if freq_text == "1" or freq_text == "1 ":
                            frequency = "Once daily (OD)"
                        elif re.match(r'\d+-\d+-\d+', freq_text):
                            frequency = "Three times daily (TDS)"
                        else:
                            frequency = freq_text
                        break
                
                # Extract duration
                duration = "Not mentioned"
                duration_match = re.search(r'(\d+)\s*(days?|weeks?|months?)\b', context)
                if duration_match:
                    duration = f"{duration_match.group(1)} {duration_match.group(2)}"
                
                # Extract route - also check the original match for route indicator
                route = "Not mentioned"
                # First check if route was in the original pattern match
                if len(match.groups()) > 0:
                    first_group = match.group(0).lower()
                    if any(word in first_group for word in ['tab', 'tablet']):
                        route = "Tablet"
                    elif any(word in first_group for word in ['cap', 'capsule']):
                        route = "Capsule"
                    elif any(word in first_group for word in ['syp', 'syrup', 'sup']):  # Handle OCR error "sup"
                        route = "Syrup"
                    elif any(word in first_group for word in ['inj', 'injection']):
                        route = "Injection"
                
                # If route not found in match, search context
                if route == "Not mentioned":
                    route_patterns = [
                        r'\b(tablet|tablets|tab|tabs)\b',
                        r'\b(capsule|capsules|cap|caps)\b',
                        r'\b(syrup|syrups|syp|sup)\b',  # Handle OCR error "sup" for "syp"
                        r'\b(injection|injections|inj)\b',
                    ]
                    
                    for route_pattern in route_patterns:
                        route_match = re.search(route_pattern, context)
                        if route_match:
                            route_word = route_match.group(0).lower()
                            if route_word in ['tablet', 'tablets', 'tab', 'tabs']:
                                route = "Tablet"
                            elif route_word in ['capsule', 'capsules', 'cap', 'caps']:
                                route = "Capsule"
                            elif route_word in ['syrup', 'syrups', 'syp', 'sup']:  # Handle OCR error
                                route = "Syrup"
                            elif route_word in ['injection', 'injections', 'inj']:
                                route = "Injection"
                            break
                
                # Use evidence word (original OCR text)
                evidence = evidence_word
                
                # Create medicine entry
                med_key = medicine_name.lower() if isinstance(medicine_name, str) and not medicine_name.startswith("Unclear") else evidence_word
                if med_key not in found_medicines:
                    found_medicines[med_key] = {
                        "name": medicine_name,
                        "strength": strength,
                        "frequency": frequency,
                        "duration": duration,
                        "route": route,
                        "evidence": evidence
                    }
    
    # Convert to list
    medicines_list = list(found_medicines.values())
    
    # Additional pass to detect standalone capitalized medicine names that might have been missed
    # This helps catch medicines like "ATENOLOL", "CIPROFLOXACIN", etc.
    capitalized_medicine_pattern = r'\b([A-Z]{4,}[0-9]*)\b'
    capitalized_matches = re.finditer(capitalized_medicine_pattern, ocr_text)
    for match in capitalized_matches:
        med_name = match.group(1)
        if med_name.lower() not in skip_words and len(med_name) >= 4:
            # Check if this medicine is already in our list
            already_found = False
            for existing_med in medicines_list:
                if existing_med["name"].lower() == med_name.lower():
                    already_found = True
                    break
            
            if not already_found:
                medicines_list.append({
                    "name": med_name,
                    "strength": "Not mentioned",
                    "frequency": "Not mentioned",
                    "duration": "Not mentioned",
                    "route": "Not mentioned",
                    "evidence": med_name
                })
    
    # If no medicines found but text suggests medications, add a general entry
    if not medicines_list and re.search(r'\b(?:medication|prescription|rx|tablet|capsule|syrup|take|dose|mg|ml|g)\b', text_lower):
        medicines_list = [{
            "name": "Medication Not Clearly Identified",
            "strength": "See prescription text",
            "frequency": "As prescribed",
            "duration": "Not mentioned",
            "route": "Not mentioned",
            "evidence": "Text suggests presence of medications"
        }]
    
    return {"medicines": medicines_list}

def validate_medicines(result):
    """
    Hard validation filter to block fake/hallucinated medicines.
    Rejects medicines without evidence or with empty names.
    """
    if "medicines" not in result:
        return {"medicines": []}
    
    validated = []
    
    for med in result["medicines"]:
        # Reject medicines without evidence (hallucinated)
        if not med.get("evidence"):
            continue
        
        # Reject medicines with empty names
        if not med.get("name") or med["name"].strip() == "":
            continue
        
        validated.append(med)
    
    return {"medicines": validated}

def process_prescription_with_enhanced_ocr(image_path, output_dir=None, languages=None):
    """Process a prescription image with enhanced OCR techniques"""
    # Use absolute path so OCR engines can load the file reliably
    image_path = os.path.abspath(image_path) if isinstance(image_path, str) else image_path
    if isinstance(image_path, str) and not os.path.isfile(image_path):
        logger.error("Prescription image not found: %s", image_path)
        return {"error": "Image file not found", "raw_text": "", "cleaned_text": "", "medications": [], "medicines_strict": {"medicines": []}}

    # First check if this image has been trained before
    try:
        from .image_trainer import ImageTrainer
        trainer = ImageTrainer()
        trained_result = trainer.find_match(image_path)
        if trained_result:
            # Return the trained result directly
            return trained_result
    except Exception as e:
        logger.debug("Error checking for trained image: %s", e)

    try:
        # Prepare output paths
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.basename(image_path)
            base_name_no_ext = os.path.splitext(base_name)[0]
            enhanced_path = os.path.join(output_dir, f"enhanced_{base_name}")
            results_path = os.path.join(output_dir, f"{base_name_no_ext}_results.json")
        else:
            enhanced_path = None
            results_path = None
        
        # STEP 1: Apply image preprocessing (grayscale, denoise, threshold, resize, morphology)
        image_data = preprocess_image(image_path)
        if not image_data:
            logger.error("Preprocessing failed for %s", image_path)
            return {
                "error": "Failed to preprocess image",
                "raw_text": "",
                "cleaned_text": "",
                "medications": [],
                "dosages": [],
                "frequencies": [],
                "routes": []
            }
        
        language_codes = _sanitize_language_codes(languages)
        runtime_settings = _load_runtime_settings()
        ocr_mode = (runtime_settings.get('mode') or 'auto').lower()
        quality_threshold = _coerce_setting(
            runtime_settings.get('quality_threshold'),
            HANDWRITING_QUALITY_THRESHOLD,
        )
        medical_signal_threshold = _coerce_setting(
            runtime_settings.get('medical_signal_threshold'),
            MEDICAL_SIGNAL_THRESHOLD,
        )
        selection_margin = _coerce_setting(
            runtime_settings.get('selection_margin'),
            HANDWRITING_SELECTION_MARGIN,
        )
        min_score = _coerce_setting(
            runtime_settings.get('min_score'),
            HANDWRITING_MIN_SCORE,
        )

        # STEP 2: Run multiple OCR passes with different preprocessing (Tesseract)
        ocr_results = run_multiple_ocr_passes(image_data, language_codes=language_codes)
        
        # STEP 3: Combine text from all OCR passes
        raw_text, confidence = combine_ocr_results(ocr_results)
        combined_score, text_quality, medical_signal = evaluate_text_candidate(raw_text)
        if not raw_text:
            logger.info("Tesseract returned no text; will try EasyOCR then PaddleOCR for handwriting.")
        if ocr_mode == 'force_handwriting':
            raw_text = ""
            confidence = 0.0
            combined_score = 0.0
            text_quality = 0.0
            medical_signal = 0.0

        handwriting_attempted = False
        handwriting_used = False
        paddleocr_attempted = False
        paddleocr_used = False

        force_handwriting = ocr_mode == 'force_handwriting'
        force_tesseract = ocr_mode == 'force_tesseract'
        force_paddleocr = ocr_mode == 'force_paddleocr'

        # When Tesseract returns empty (e.g. handwritten prescription), try EasyOCR first
        # then PaddleOCR. EasyOCR is generally better for handwriting.
        should_try_handwriting = (
            force_handwriting
            or (not force_tesseract and not force_paddleocr and (
                not raw_text
                or text_quality < quality_threshold
                or combined_score < min_score
                or medical_signal < medical_signal_threshold
            ))
        )
        should_try_paddle = (
            force_paddleocr
            or (not raw_text or text_quality < quality_threshold or combined_score < min_score)
        )

        # Try EasyOCR first when we have no or poor text (handwritten-friendly)
        if should_try_handwriting and easyocr is not None:
            handwriting_attempted = True
            handwriting_text, handwriting_confidence = run_handwriting_ocr(image_path, language_codes)
            handwriting_score, handwriting_quality, handwriting_signal = evaluate_text_candidate(handwriting_text)

            should_switch = False
            if handwriting_text:
                if not raw_text:
                    should_switch = True
                elif handwriting_score >= combined_score + selection_margin:
                    should_switch = True
                elif combined_score < min_score and handwriting_score > combined_score:
                    should_switch = True
                elif medical_signal < medical_signal_threshold and handwriting_signal > medical_signal:
                    should_switch = True

            if should_switch:
                raw_text = handwriting_text
                confidence = handwriting_confidence
                text_quality = handwriting_quality
                medical_signal = handwriting_signal
                combined_score = handwriting_score
                handwriting_used = True
                if handwriting_text:
                    logger.info("Using EasyOCR (handwriting) result: %d chars", len(handwriting_text))
        elif force_handwriting:
            handwriting_attempted = True

        # Then try PaddleOCR if still no/poor text
        if should_try_paddle and PADDLEOCR_AVAILABLE and (not raw_text or not handwriting_used):
            paddleocr_attempted = True
            paddleocr_text, paddleocr_confidence = run_paddleocr(image_path, language_codes)
            paddleocr_score, paddleocr_quality, paddleocr_signal = evaluate_text_candidate(paddleocr_text)

            should_switch_paddle = False
            if paddleocr_text:
                if not raw_text:
                    should_switch_paddle = True
                elif paddleocr_score >= combined_score + selection_margin:
                    should_switch_paddle = True
                elif combined_score < min_score and paddleocr_score > combined_score:
                    should_switch_paddle = True
                elif medical_signal < medical_signal_threshold and paddleocr_signal > medical_signal:
                    should_switch_paddle = True

            if should_switch_paddle:
                raw_text = paddleocr_text
                confidence = paddleocr_confidence
                text_quality = paddleocr_quality
                medical_signal = paddleocr_signal
                combined_score = paddleocr_score
                paddleocr_used = True
                if paddleocr_text:
                    logger.info("Using PaddleOCR result: %d chars", len(paddleocr_text))

        # Final fallback: if still no text, try EasyOCR once more with English only (handwriting)
        if not raw_text and easyocr is not None:
            logger.info("Final fallback: trying EasyOCR with English only.")
            fallback_text, fallback_conf = run_handwriting_ocr(image_path, language_codes=["eng"])
            if fallback_text:
                raw_text = fallback_text
                confidence = fallback_conf
                handwriting_used = True
                logger.info("Final fallback EasyOCR extracted %d chars.", len(fallback_text))
        
        # Return a basic structure even if text extraction fails
        # This will allow trained images to still work
        if not raw_text:
            strict_medicines = extract_medicines_strict("")  # Empty text returns empty array
            strict_medicines = validate_medicines(strict_medicines)  # Apply validation
            return {
                "image_path": image_path,
                "preprocessed_image": image_data.get("enhanced", ""),
                "raw_text": "",
                "cleaned_text": "",
                "medications": [],
                "dosages": [],
                "frequencies": [],
                "routes": [],
                "durations": [],
                "medicines_strict": strict_medicines,
                "confidence": float(confidence) * 100 if confidence else 50.0,
                "languages_used": language_codes,
                "ocr_engine": "paddleocr" if paddleocr_used else ("handwriting" if handwriting_used else "tesseract"),
                "paddleocr_attempted": paddleocr_attempted,
                "paddleocr_used": paddleocr_used,
                "handwriting_attempted": handwriting_attempted,
                "handwriting_detected": handwriting_used,
                "text_quality": text_quality,
                "medical_signal": medical_signal,
                "ocr_mode": ocr_mode,
            }
        
        # STEP 4: Apply medical dictionary correction
        corrected_text = apply_medical_dictionary_correction(raw_text)
        
        # STEP 5: Extract medical entities using STRICT extraction rules
        strict_medicines = extract_medicines_strict(corrected_text)
        
        # STEP 6: Apply hard validation filter to block fake/hallucinated medicines
        strict_medicines = validate_medicines(strict_medicines)
        
        
        # Convert strict format to backward-compatible format
        medicines_list = strict_medicines.get("medicines", [])
        medications = []
        dosages = []
        frequencies = []
        routes = []
        durations = []
        
        for med in medicines_list:
            medications.append(med.get("name", "Not mentioned"))
            dosages.append(med.get("strength", "Not mentioned"))
            frequencies.append(med.get("frequency", "Not mentioned"))
            routes.append(med.get("route", "Not mentioned"))
            durations.append(med.get("duration", "Not mentioned"))
        
        # Build the results
        results = {
            "image_path": image_path,
            "preprocessed_image": image_data["enhanced"],
            "raw_text": raw_text,
            "cleaned_text": corrected_text,
            "medications": medications,
            "dosages": dosages,
            "frequencies": frequencies,
            "routes": routes,
            "durations": durations,
            "medicines_strict": strict_medicines,  # Include strict format for new code
            "confidence": float(confidence) * 100 if confidence else 90.0,
            "languages_used": language_codes,
            "ocr_engine": "paddleocr" if paddleocr_used else ("handwriting" if handwriting_used else "tesseract"),
            "paddleocr_attempted": paddleocr_attempted,
            "paddleocr_used": paddleocr_used,
            "handwriting_detected": handwriting_used,
            "handwriting_attempted": handwriting_attempted,
            "text_quality": text_quality,
            "medical_signal": medical_signal,
            "ocr_mode": ocr_mode,
        }
        
        # Save results to file if output_dir is provided
        if results_path:
            try:
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=4)
            except Exception as e:
                print(f"Error saving results: {str(e)}")
        
        return results
        
    except Exception as e:
        print(f"Error in OCR processing: {str(e)}")
        strict_medicines = extract_medicines_strict("")  # Empty text returns empty array
        strict_medicines = validate_medicines(strict_medicines)  # Apply validation
        return {
            "error": f"Processing error: {str(e)}",
            "raw_text": "",
            "cleaned_text": "",
            "medications": [],
            "dosages": [],
            "frequencies": [],
            "routes": [],
            "durations": [],
            "medicines_strict": strict_medicines,
            "languages_used": _sanitize_language_codes(languages),
        }

# ===================================================
# Text Extraction using Tesseract OCR
# ===================================================
# PSM modes: 3=full auto, 4=single column, 5=block, 6=single block (default), 11=sparse text, 13=raw line
# OEM: 0=legacy, 1=LSTM only, 2=legacy+LSTM, 3=default (LSTM + legacy)
TESSERACT_PSM_FALLBACK = [6, 11, 4, 3, 13]
TESSERACT_OEM = 3


def extract_text_tesseract(image, config=None, language_codes=None):
    """
    Extract text using Pytesseract with production-ready config and fallbacks.

    - Uses --oem 3 (LSTM + legacy) and --psm 6 (single block) by default.
    - If config is provided, uses it; otherwise builds config from language_codes.
    - Tries multiple PSM modes if the first attempt returns empty or fails.
    - Accepts: file path (str), numpy array (BGR or grayscale).
    """
    def _run_ocr(img_input, cfg: str) -> str:
        if isinstance(img_input, np.ndarray):
            if len(img_input.shape) == 3:
                img_pil = Image.fromarray(cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB))
            else:
                img_pil = Image.fromarray(img_input)
            return pytesseract.image_to_string(img_pil, config=cfg)
        if isinstance(img_input, str) and os.path.exists(img_input):
            return pytesseract.image_to_string(img_input, config=cfg)
        return ""

    effective_config = config
    if effective_config is None:
        configs = build_tesseract_configs(language_codes)
        effective_config = configs[0][0] if configs else f"--oem {TESSERACT_OEM} --psm 6"

    try:
        text = _run_ocr(image, effective_config)
        text = (text or "").strip()
        if text:
            logger.debug("Tesseract extracted %d chars with config: %s", len(text), effective_config)
            return text
    except pytesseract.TesseractError as e:
        logger.warning("Tesseract error (trying PSM fallbacks): %s", e)
    except Exception as e:
        logger.warning("Pytesseract exception: %s", e)

    # Fallback: try other PSM modes with same language
    lang_part = "-l eng"
    if language_codes is not None:
        codes = _sanitize_language_codes(language_codes)
        if codes:
            lang_part = "-l " + "+".join(codes)
    for psm in TESSERACT_PSM_FALLBACK[1:]:
        try:
            fallback_config = f"--oem {TESSERACT_OEM} --psm {psm} {lang_part}"
            text = _run_ocr(image, fallback_config)
            text = (text or "").strip()
            if text:
                logger.info("Tesseract succeeded with PSM %s", psm)
                return text
        except Exception:
            continue
    logger.debug("Tesseract returned no text for image")
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
    results = process_prescription_with_enhanced_ocr(args.image, args.output)
    
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
        
        print(f"\nConfidence: {results['confidence']:.1f}%")
        
        print("\nProcessing complete!")
