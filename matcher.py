# matcher.py
import re
import unicodedata
import fasttext
import xgboost
import joblib
from tqdm import tqdm
from collections import defaultdict
from rapidfuzz.fuzz import partial_ratio
import os
from urllib.request import urlretrieve

MODEL_PATH = 'models/lid.176.bin'

if not os.path.exists(MODEL_PATH):
    os.makedirs('models', exist_ok=True)
    urlretrieve(
        'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin',
        MODEL_PATH
    )

lang_model = fasttext.load_model(MODEL_PATH)

# Load fastText language detection model
lang_model = fasttext.load_model('models/lid.176.bin')

# Load trained XGBoost model
xgb_model = joblib.load("models/model.pkl")

# Normalize text: lower, strip accents/punctuation
def normalize(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

# Fuzzy match using normalized text
def fast_match(name1, name2):
    return partial_ratio(normalize(name1), normalize(name2)) / 100.0

# Language detection using fastText
def detect_lang(text):
    try:
        label, confidence = lang_model.predict(text.strip().replace('\n', ''))
        return label[0].replace('__label__', ''), float(confidence[0])
    except:
        return 'unknown', 0.0

# Main matcher function for API
def match_supplier_to_reference(supplier_rooms, reference_rooms):
    results = []
    for sup in supplier_rooms:
        lang_sup, conf_sup = detect_lang(sup['supplierRoomName'])

        for ref in reference_rooms:
            lang_ref, conf_ref = detect_lang(ref['roomName'])
            fuzzy_score = fast_match(sup['supplierRoomName'], ref['roomName'])

            features = {
                "lp_id_match": 0,  # can be extended
                "hotel_id_match": 0,  # can be extended
                "room_id_match": int(sup['supplierRoomId'] == ref['roomId']),
                "fuzzy_score": fuzzy_score,
            }

            proba = xgb_model.predict_proba([[features[k] for k in features]])[0][1]

            results.append({
                "supplierRoomId": sup['supplierRoomId'],
                "supplierRoomName": sup['supplierRoomName'],
                "refRoomId": ref['roomId'],
                "refRoomName": ref['roomName'],
                "match_score": float(round(proba, 4)),
                "fuzzy_score": float(round(fuzzy_score, 4)),
                "lang_supplier": lang_sup,
                "lang_ref": lang_ref,
                "lang_conf_supplier": float(round(conf_sup, 4)),
                "lang_conf_ref": float(round(conf_ref, 4))
            })
    return results

