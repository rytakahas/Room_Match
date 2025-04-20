# Room_Match

Cupid API’s Room Match

##  Objective: Multilingual Room Matching with Fuzzy Logic and XGBoost

Build a machine learning API similar to Cupid's Room Match API.
This API handles POST requests and returns room match predictions between suppliers and reference rooms.
Supports mixed-language input (e.g., English + Arabic + Korean).

---

###  Project Structure

```bash
Room_Match/
├── README.md                  # This file
├── requirements.txt           # All dependencies
├── app.py                     # Flask API server
├── matcher.py                 # Core logic for matching
├── models/                    # Trained XGBoost model + fastText model
│   ├── model.pkl
│   └── lid.176.bin
├── sample_request.json        # Example POST request payload
├── test_post.py               # Simple script to send test POST request
├── notebooks/
│   └── room_match_dev.ipynb   # EDA, model training and evaluation
└── __pycache__/
```

---

##  How to Run the API

###  Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

###  Step 2: Start the Flask API
```bash
FLASK_APP=app.py flask run --host=0.0.0.0 --port=5050
```

###  Step 3: Send a test request
```bash
curl -X POST http://127.0.0.1:5050/room_match \
  -H 'Content-Type: application/json' \
  -d @sample_request.json
```

Or run:
```bash
python test_post.py
```

---

##  Input Format (sample_request.json)
```json
{
  "inputCatalog": [
    {
      "supplierId": "nuitee",
      "supplierRoomInfo": [
        {"supplierRoomId": "2", "supplierRoomName": "Classic Room - Olympic Queen Bed - ROOM ONLY"}
      ]
    }
  ],
  "referenceCatalog": [
    {
      "propertyId": "5122906",
      "propertyName": "Pestana Park Avenue",
      "referenceRoomInfo": [
        {"roomId": "512290602", "roomName": "Classic Room"},
        {"roomId": "512290608", "roomName": "Classic Room - Disability Access"}
      ]
    }
  ]
}
```

---

##  Matching Logic

- Language detection via fastText (`lid.176.bin`)
- Text normalization (lowercase, strip accents, remove punctuation)
- Fuzzy match using `rapidfuzz.partial_ratio`
- Feature engineering:
  - `room_id_match`
  - `fuzzy_score`
- Model: XGBoost classifier with Optuna tuning

```python
features = ["lp_id_match", "hotel_id_match", "room_id_match", "fuzzy_score"]
label = int(fuzzy_score >= 0.85)
```

###  Optional Upgrade: SentenceTransformer
If GPU (e.g., T4 in Colab) is available:
```python
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
```
This improves multilingual and mixed-language understanding.

---

##  Model Training Pipeline

1. **Data Cleaning**
   - Drop rows with missing names
   - Normalize strings
   - Filter candidate pairs by ID

2. **Label Generation**
   - `label = 1` if strong ID match and fuzzy_score >= 0.85 

3. **Model**
   - XGBoost classifier with Optuna
   - Metrics: F1, ROC-AUC, Confusion matrix

4. **Saved Output**
   - `models/model.pkl`: trained classifier
   - `models/lid.176.bin`: fastText language detection

---

##  Example Output
```json
{
  "supplierRoomId": "2",
  "supplierRoomName": "Classic Room - Olympic Queen Bed - ROOM ONLY",
  "refRoomId": "512290602",
  "refRoomName": "Classic Room",
  "fuzzy_score": 1.0,
  "match_score": 0.9991,
  "lang_supplier": "en",
  "lang_ref": "en"
}
```

---

## Contact
For questions, ideas, or improvements, feel free to open an issue or pull request.
