# Room_Match
Cupid API’s Room Match


    Room-Match/
    ├── README.md                # Overview, setup instructions, API usage
    ├── requirements.txt         # All dependencies
    ├── notebook/                # EDA, model training, evaluation
    │       └── room_match_dev.ipynb
    ├── src/                     # Core backend code
    │   ├── main.py              # FastAPI app
    │   ├── api/                 # API endpoint logic
    │   │   └── room_match.py
    │   ├── models/              # Trained model files or loading code
    │   │   └── model.pkl
    │   ├── utils/               # Preprocessing, tokenization, helpers
    │   │   ├── preprocess.py
    │   │   └── features.py
    │   └── inference.py         # Predict function used by API
    ├── tests/                   # Unit and integration tests
    │   └── test_api.py

    
    
  

### Objective: Classification for Wine Quality (Binary Classification)

Build a machine learning API similar to the Cupid API’s Room Match feature. <br> 
The API should handle POST requests and return sample request/response payloads in a similar <br> 
format to the Cupid Room Match API. Provide a detailed explanation of your development process, <br> 
including how you collect and process data, develop models, and scale the system.


**Random Forest** and **XGBoost** <br> 
The workflow includes data preprocessing, model training, <br>
hyperparameter optimization, evaluation, and visualization of the results.

#### Steps
1: Data Exploration and Preprocessing <br>
2: Model Training with Random Forest and XGBoost <br>
3: Evaluation Metrics and Visualization <br>
4: Deliverables <br>
<br>
<br>

1. **Data Preparation**
    - Load the wine quality dataset.
    - Analyze statistics and correlations of features.
    - Transform multiple classifications of wine quality
     to binary classification.
    - Standard Scaling ($\mu$ = 0, $\sigma$ = 1)

2. **Model Training**
    - Split the dataset into training and testing sets.
    - Train with Random Forest (w/o grid search) 
    and XGBoost with optuna.

3. **Evaluation Metrics and Visualization**
    - Evaluate precision, recall, and F1 scores
    - Visualized ROC curve, confusion matrix, and
    feature importance 


4. **Deliverables**
    - RF, and XGBoost trained models were saved to **pkl**
    files, and reproducing test results

