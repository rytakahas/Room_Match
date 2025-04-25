# matcher.py
import re
import unicodedata
import torch
import joblib
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util

# Load SentenceTransformer model (E5 multilingual)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer("intfloat/multilingual-e5-base", device=device)

# Load trained XGBoost model
xgb_model = joblib.load("models/model.pkl")

# =========================
# Text Normalization
# =========================
def normalize(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# =========================
# Embedding Generator
# =========================
def embed_text(text):
    norm_text = normalize(text)
    embedding = model.encode(norm_text, convert_to_tensor=True, normalize_embeddings=True, device=device)
    return embedding

# =========================
# Main Matcher Function
# =========================
def match_supplier_to_reference(supplier_rooms, reference_rooms):
    results = []
    
    print("Generating embeddings for reference rooms...")
    ref_embeddings = {}
    for ref in reference_rooms:
        ref['normalized'] = normalize(ref['roomName'])
        ref['embedding'] = embed_text(ref['roomName'])
        ref_embeddings[ref['roomId']] = ref['embedding']
    
    print("Matching supplier rooms...")
    for sup in tqdm(supplier_rooms):
        sup_embedding = embed_text(sup['supplierRoomName'])

        for ref in reference_rooms:
            cosine_sim = float(util.cos_sim(sup_embedding, ref['embedding'])[0][0])

            features = {
                "lp_id_match": 0,  # Extend if available
                "hotel_id_match": 0,  # Extend if available
                "room_id_match": int(sup['supplierRoomId'] == ref['roomId']),
                "cosine_sim": cosine_sim,
            }

            proba = xgb_model.predict_proba([[features[k] for k in features]])[0][1]

            results.append({
                "supplierRoomId": sup['supplierRoomId'],
                "supplierRoomName": sup['supplierRoomName'],
                "refRoomId": ref['roomId'],
                "refRoomName": ref['roomName'],
                "match_score": float(round(proba, 4)),
                "cosine_sim": float(round(cosine_sim, 4))
            })

    return results

