# Room Match API

A Flask API for matching multilingual hotel room names using SentenceTransformer matching and language detection.

---

## Setup

### Step 1: Install dependencies
Make sure you have Python 3.10+ installed, then run:

```bash
pip install -r requirements.txt
```

---

### Step 2: Start the Flask API (manual)

```bash
FLASK_APP=app.py flask run --host=0.0.0.0 --port=5050
```

---

### Step 3: Send a test request manually (if not using the script)

```bash
curl -X POST http://127.0.0.1:5050/room_match \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

Or use Python:

```bash
python test_post.py
```


Or just run the helper script which starts the API and runs tests:

```bash
./run_server_and_test.sh sample_request.json
```

> This script installs dependencies, starts the server, and sends a test POST request using `sample_request.json`

---

## Project Structure

```
Room_Match/
├── README.md                  # This file
├── requirements.txt           # All dependencies
├── app.py                     # Flask API server
├── matcher.py                 # Core logic for matching
├── models/                    # Trained XGBoost model + fastText model
│   ├── model.pkl              # - Generate in the notebook: room_match_dev.ipynb
│   └── lid.176.bin            # - Download: https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
├── sample_request.json        # Example POST request payload
├── test_post.py               # Simple script to send test POST request
├── run_server_and_test.sh     # Script to run server and test it
├── scripts/                   # Utility or dev scripts
│   └── __init__.py            # Placeholder
├── tests/                     # Unit tests for matcher and API
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_room_match.py
│   └── test_post.py
├── notebooks/
│   └── room_match_dev.ipynb   # EDA, model training and evaluation
└── __pycache__/
```

---

## License
MIT


