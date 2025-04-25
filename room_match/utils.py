import os
import re
import unicodedata
from collections import defaultdict
from rapidfuzz.fuzz import partial_ratio
import fasttext

# File paths
FASTTEXT_MODEL_PATH = 'models/lid.176.bin'
FASTTEXT_MODEL_URL = 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin'

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

# Download FastText language model if missing
if not os.path.exists(FASTTEXT_MODEL_PATH):
    print("Downloading FastText language model...")
    urlretrieve(FASTTEXT_MODEL_URL, FASTTEXT_MODEL_PATH)

# Load the model
lang_model = fasttext.load_model(FASTTEXT_MODEL_PATH)

# You can add similar logic for model.pkl if needed:
# MODEL_PKL_PATH = "models/model.pkl"
# MODEL_PKL_URL = "https://your-url.com/model.pkl"
# if not os.path.exists(MODEL_PKL_PATH):
#     print("Downloading model.pkl...")
#     urlretrieve(MODEL_PKL_URL, MODEL_PKL_PATH)

# Load fastText language detection model
lang_model = fasttext.load_model('models/lid.176.bin')  # adjust path if needed

def detect_lang(text):
    """Detects language using fastText model."""
    try:
        label, confidence = lang_model.predict(text.strip().replace('\n', ''))
        return label[0].replace('__label__', ''), confidence[0]
    except:
        return 'unknown', 0.0

def normalize(text):
    """Lowercase and remove accents, punctuation, and whitespace."""
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')  # Remove accents
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.strip()

def fast_match(name1, name2):
    """Fuzzy match between two room names using normalized text."""
    return partial_ratio(normalize(name1), normalize(name2)) / 100.0

def generate_matches(df_rooms, df_ref):
    """Main multilingual fuzzy match pipeline between rooms and references."""
    ref_buckets = defaultdict(list)
    for ref in df_ref.itertuples(index=False):
        ref_buckets[f"lp_{ref.lp_id}"].append(ref)
        ref_buckets[f"h_{ref.hotel_id}"].append(ref)
        ref_buckets[f"rid_{ref.room_id}"].append(ref)

    match_rows = []

    for r in df_rooms.itertuples(index=False):
        if not isinstance(r.supplier_room_name, str):
            continue

        lang_supplier, conf_supplier = detect_lang(r.supplier_room_name)

        candidates = (
            ref_buckets.get(f"lp_{r.lp_id}", []) +
            ref_buckets.get(f"h_{r.core_hotel_id}", []) +
            ref_buckets.get(f"rid_{r.core_room_id}", []) +
            ref_buckets.get(f"rid_{r.supplier_room_id}", [])
        )

        if not candidates:
            continue

        for ref in candidates:
            if not isinstance(ref.room_name, str):
                continue

            sim = fast_match(r.supplier_room_name, ref.room_name)
            lang_ref, conf_ref = detect_lang(ref.room_name)

            hotel_match = int(r.core_hotel_id == ref.hotel_id)
            if not hotel_match:
                continue

            lp_match = int(r.lp_id == ref.lp_id)
            room_match = int((r.core_room_id == ref.room_id) or (r.supplier_room_id == ref.room_id))

            match_rows.append({
                'core_room_id': r.core_room_id,
                'supplier_room_id': r.supplier_room_id,
                'ref_room_id': ref.room_id,
                'lp_id_match': lp_match,
                'hotel_id_match': hotel_match,
                'room_id_match': room_match,
                'fuzzy_score': sim,
                'label': int(sim >= 0.85),
                'supplier_room_name': r.supplier_room_name,
                'ref_room_name': ref.room_name,
                'lang_supplier': lang_supplier,
                'lang_ref': lang_ref,
                'lang_conf_supplier': conf_supplier,
                'lang_conf_ref': conf_ref
            })

    return match_rows

