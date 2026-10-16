# ArdayScore — Student Performance Prediction

ArdayScore is an end-to-end machine learning project that predicts whether a student is likely to **Pass** or **Fail** based on daily study hours and attendance percentage.

The project combines a trained scikit-learn classification pipeline, a FastAPI prediction service, and a responsive web client. It is designed as a practical ML portfolio project that demonstrates the journey from data preparation and model training to API integration and a usable interface.

## Features

- Logistic Regression classification pipeline
- Input preprocessing with imputation and feature scaling
- Hyperparameter tuning with `GridSearchCV` and stratified cross-validation
- FastAPI REST endpoint with Pydantic validation
- Confidence score returned with every prediction
- Responsive English web interface
- Separate frontend and backend applications
- CORS configuration for local frontend development

## How it works

1. `train_model.py` loads `student.csv` and selects `studyhour`, `attendance`, and `result`.
2. The training workflow splits the data using stratification, preprocesses numeric features, and tunes Logistic Regression parameters.
3. The best pipeline is evaluated and saved as `model.pkl`.
4. `main.py` loads the saved pipeline and exposes `POST /predict`.
5. The client sends student inputs to the API and displays the predicted result and model confidence.

## Project structure

```text
.
├── client/
│   ├── index.html       # Responsive web client
│   └── README.md        # Client run instructions
├── main.py              # FastAPI application
├── train_model.py       # Model training and evaluation workflow
├── student.csv          # Example training data
├── model.pkl            # Saved scikit-learn pipeline
├── requirements.txt     # Python dependencies
└── .gitignore
```

## Run locally

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 3. Start the backend

From the project root:

```powershell
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at `http://127.0.0.1:8000` and interactive documentation is available at `http://127.0.0.1:8000/docs`.

### 4. Start the frontend

Open a second terminal:

```powershell
cd client
python -m http.server 5173 --bind 127.0.0.1
```

Open `http://127.0.0.1:5173` in your browser.

## API usage

### Request

```http
POST /predict
Content-Type: application/json
```

```json
{
  "studyhour": 5.5,
  "attendance": 75
}
```

### Response

```json
{
  "prediction": "Pass",
  "confidence": 0.8751
}
```

## ML approach

The model uses two numeric features:

- `studyhour`: average study hours per day
- `attendance`: attendance percentage

The pipeline applies median imputation and standardization before Logistic Regression. `GridSearchCV` evaluates regularization strength and class weighting using five-fold stratified cross-validation and macro F1 scoring.

## Portfolio value

This project demonstrates practical machine learning engineering skills across the full workflow:

- Data preparation and feature selection
- Reproducible model training and evaluation
- Scikit-learn pipelines and hyperparameter search
- Model serialization and inference
- REST API design and validation
- Frontend-to-backend integration
- Clear separation between application layers

## Future improvements

- Add a larger and more representative dataset
- Add automated tests and CI with GitHub Actions
- Track experiments and model versions
- Add model explainability with feature contribution visualizations
- Deploy the API and client to production hosting

## Disclaimer

ArdayScore is an educational portfolio project. Its predictions are estimates based on a small example dataset and should not be used as the sole basis for academic decisions.
