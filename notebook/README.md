## Multilingual Room Matching with SentenceTransformer (multilingual-e5-base) and XGBoost

This project builds a multilingual, fuzzy logic–based , and SentenceTransformer machine learning pipelines for **matching hotel room listings** between suppliers and a reference dataset. It supports **multiple and mixed languages** (e.g., English, Korean, Arabic) using `fastText` and `rapidfuzz`, followed by a binary classification model using **XGBoost**.

---

### Requirements

Create a `requirements.txt` file with:

pandas tqdm numpy rapidfuzz xgboost scikit-learn matplotlib seaborn fasttext sentence-transformers torch unicodedata2



#### Install via pip

```bash
pip install -r requirements.txt
```
- Dataset
updated_core_rooms.csv → loaded into df_rooms (supplier listings)

reference_rooms-1737378184366.csv → loaded into df_ref (reference data)

```python
df_rooms = pd.read_csv("updated_core_rooms.csv")
df_ref = pd.read_csv("reference_rooms-1737378184366.csv")
```
### Exploratory Data Analysis (EDA)

Dropped rows with missing supplier_room_name or room_name

Verified data types: room_id, core_room_id, hotel_id, etc.

Removed rows with NaN, empty strings, or duplicate entries

### Data Preparation Strategy
#### Matching Logic
- **`lp_id` (Listing Platform ID)** identifies the supplier (e.g., Booking.com, Agoda),  
  but **does not uniquely define a hotel or room**.

- The **most reliable match** occurs when both `hotel_id` and `room_id` match —  
  this strongly indicates the same room, regardless of `lp_id`.

- Matching only on `lp_id` is **weak**, since the same supplier can list multiple rooms/hotels.

- **Safe Rule**: If `hotel_id` and `room_id` match → consider it the **same room**.  
  Use `lp_id` only as a **secondary signal**, not for strict filtering.
#### Matching Logic Summary

| hotel_id Match | room_id Match | lp_id Match | Is It the Same Room? | Explanation                         |
|----------------|----------------|-------------|-----------------------|-------------------------------------|
| Yes          | Yes          |(Yes / No) Either | Yes                | Strong match – same room in same hotel |
| Yes          | No           | (Yes / No)Either | No                 | Same hotel, different rooms         |
| No           | (Yes / No) Any      | Yes         | No                 | Different hotels → no match         |

2. Room Name Matching
Using multilingual support:

### Language Detection: fastText lid.176.bin

  - Fuzzy Token Matching: rapidfuzz.partial_ratio (normalized, punctuation-free comparison)

  - Handles mixed-language names like "Deluxe Room (디럭스 패밀리 트윈)"
  - **Limitation**: `rapidfuzz.partial_ratio` is effective for within-language string comparisons but may return unreliable similarity scores for cross-language tokens (e.g., English vs. Korean). This is due to its reliance on character-level matching rather than semantic understanding.  
  → For improved multilingual matching, consider using embedding-based approaches (e.g., fastText vectors or multilingual transformers).

### Recommended for improved multilingual matching:
  - Use embedding-based similarity with SentenceTransformer and the model intfloat/multilingual-e5-base, which captures cross-lingual semantic relationships more effectively than token-based methods like rapidfuzz.

### Matching Logic
Each candidate pair is labeled:


Condition	Feature
hotel_id, room_id match	Feature flags
fuzzy_score >= 0.85	Considered a match
cosine_sim >= 0.85 Considered a match
Label = 1 if any strong match is present	Binary label

```python
label = int(id_match (hotel_id & room_id) and fuzzy_score >= 0.85)
```

#### Multilingual matching: Deep Sentence Embedding for Multilingual Matching
If GPU (e.g., Colab T4 or local CUDA) is available, you can boost multilingual matching 
accuracy using SentenceTransformer for semantic similarity:

```python
from sentence_transformers import SentenceTransformer, util
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer(""intfloat/multilingual-e5-base"", device=device)

# Compare room names:
sim = util.cos_sim(model.encode(name1, convert_to_tensor=True), 
                   model.encode(name2, convert_to_tensor=True)).item()
```

This model supports 100+ languages and works great on mixed-language room names like:

"Deluxe Twin Room, 2 Double Beds (디럭스 패밀리 트윈)"

"Deluxe Twin Room with two double beds"

This embedding-based approach is ideal for cross-lingual matching scenarios where traditional token-based methods (e.g., fuzzy matching) may fail to capture semantic similarity.

### Model: XGBoost Classification
We use XGBoost to learn from:

lp_id_match

hotel_id_match

room_id_match

cosine_sim

### Training Strategy
80/20 train-test split

Hyperparameter tuning via Optuna

Metrics:

- F1-score

- ROC-AUC

- Confusion matrix

- Probability-based ranking

```python
X = match_df[['lp_id_match', 'hotel_id_match', 'room_id_match', 'cosine_sim']]
y = match_df['label']

model = xgboost.XGBClassifier(...)
model.fit(X_train, y_train)
```

### Results
~ 100% F1-score on test set

Probabilistic predictions allow for ranked match scoring

Human-friendly sample evaluation included:

  - High-confidence matches

  - Rejected mismatches

  - Sample Predictions

### Sample Predictions

| Supplier Name             | Ref Name                              | Fuzzy Score | Prediction   |
|--------------------------|----------------------------------------|-------------|--------------|
| غرفة ديلوكس              | Deluxe Room (ディラックス ルーム)       | 0.9050      |  Match       |

