# Multilingual Room Matching with Fuzzy Logic and XGBoost

This project builds a multilingual, fuzzy logic–based machine learning pipeline for **matching hotel room listings** between suppliers and a reference dataset. It supports **multiple and mixed languages** (e.g., English, Korean, Arabic) using `fastText` and `rapidfuzz`, followed by a binary classification model using **XGBoost**.

---

## Requirements

Create a `requirements.txt` file with:

pandas tqdm numpy rapidfuzz xgboost scikit-learn matplotlib seaborn fasttext sentence-transformers torch unicodedata2

perl
Copy
Edit

### Install via pip

```bash
pip install -r requirements.txt
📂 Dataset
updated_core_rooms.csv → loaded into df_rooms (supplier listings)

reference_rooms-1737378184366.csv → loaded into df_ref (reference data)


df_rooms = pd.read_csv("updated_core_rooms.csv")
df_ref = pd.read_csv("reference_rooms-1737378184366.csv")
### Exploratory Data Analysis (EDA)
Dropped rows with missing supplier_room_name or room_name

Verified data types: room_id, core_room_id, hotel_id, etc.

Removed rows with NaN, empty strings, or duplicate entries

### Data Preparation Strategy
1. Candidate Filtering by ID
We first narrow down potential matches using:

lp_id (strong signal)

core_hotel_id and hotel_id

core_room_id, supplier_room_id, and room_id

This gives a small candidate set for each supplier room.

2. Room Name Matching
Using multilingual support:

### Language Detection: fastText lid.176.bin

### Fuzzy Token Matching: rapidfuzz.partial_ratio (normalized, punctuation-free comparison)

### Handles mixed-language names like "Deluxe Room (디럭스 패밀리 트윈)"

### Matching Logic
Each candidate pair is labeled:


Condition	Feature
lp_id, hotel_id, room_id match	Feature flags
fuzzy_score >= 0.85	Considered a match
Label = 1 if any strong match is present	Binary label
python
Copy
Edit
label = int(fuzzy_score >= 0.85 or id_match)
### Optional: Deep Sentence Embedding for Multilingual Matching
If GPU (e.g., Colab T4 or local CUDA) is available, you can boost multilingual matching accuracy using SentenceTransformer for semantic similarity:


from sentence_transformers import SentenceTransformer, util
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)

# Compare room names:
sim = util.cos_sim(model.encode(name1, convert_to_tensor=True), 
                   model.encode(name2, convert_to_tensor=True)).item()
This model supports 100+ languages and works great on mixed-language room names like:

"Deluxe Twin Room, 2 Double Beds (디럭스 패밀리 트윈)"

"Deluxe Twin Room with two double beds"

### Model: XGBoost Classification
We use XGBoost to learn from:

lp_id_match

hotel_id_match

room_id_match

fuzzy_score

### Training Strategy
80/20 train-test split

Hyperparameter tuning via Optuna

Metrics:

- F1-score

- ROC-AUC

- Confusion matrix

- Probability-based ranking

X = match_df[['lp_id_match', 'hotel_id_match', 'room_id_match', 'fuzzy_score']]
y = match_df['label']

model = xgboost.XGBClassifier(...)
model.fit(X_train, y_train)
### Results
~99.6% F1-score on test set

Probabilistic predictions allow for ranked match scoring

Human-friendly sample evaluation included:

### High-confidence matches

### Rejected mismatches

### Sample Predictions

Supplier Name	Ref Name	Fuzzy	Prediction
Deluxe Room, 2 Beds	Deluxe Room (디럭스)	0.92	: Match
Economy Bunk	Family Suite	0.41	: No Match

