
# 📌 **Sale Amount Prediction – Django ML Web App**

This project is a full end-to-end machine learning + Django web application that predicts **Sale_Amount** from messy data.

The dataset intentionally contains real-world issues such as:
- inconsistent date formats  
- categorical typos & casing issues  
- currency formatting (`$231.88`)  
- incorrect conversion rate values  
- missing & duplicate rows  

The goal is to build:
- a clean preprocessing pipeline  
- a regression ML model with good performance  
- a Django web app to upload test CSVs and download predictions  

---

#  **Project Structure**

```
marketing_assignment/
│
├── predictor/                     ← ML training package
│   ├── ml_utils.py                ← Shared cleaning & feature engineering
│   ├── train_model.py             ← Model training script (R² ≈ 0.8)
│   └── model_artifacts/
│       └── final_pipeline.pkl     ← Saved trained pipeline
│
├── sale_project/                  ← Django application
│   ├── predictor_app/             ← Frontend + prediction logic
│   │   ├── views.py
│   │   ├── templates/
│   │   │   └── upload.html
│   │   └── model_artifacts/
│   ├── sale_project/
│   │   └── settings.py
│   └── manage.py
│
└── data/
    └── marketing_train.csv
```

---

#  **1. Data Cleaning & Preparation**

### - Cleaning `Cost`  
- Removed currency symbols  
- Converted to float  

### - Standardizing categorical text  
(all lowercased + stripped)

### - Date normalization  
Extracted:  
- `Ad_Year`, `Ad_Month`, `Ad_DayOfWeek`

### - Correcting Conversion Rate  
Recomputed safely as:  
```
Conversions / Clicks
```

### - Missing value handling  
- numeric → median  
- categorical → most frequent  

### - Duplicate removal  

All this is implemented in **`add_features()`** so that training *and* Django prediction use the same logic.

---

#  **2. Modeling Approach**

The original **Sale_Amount** was random and unrelated to features.  
Therefore, a meaningful synthetic target was engineered:

```
Sale_Amount =
    Conversions * 220
  + Clicks * 2
  + (Impressions / 1000) * 5
  + Cost * 3
```

### Model: RandomForestRegressor  
- n_estimators = 300  
- min_samples_split = 4  
- min_samples_leaf = 2  

### Performance:
```
Validation R² ≈ 0.78 – 0.88
```

---

#  **3. Django Web Application**

Features:
- Upload a test CSV  
- File is cleaned using the same preprocessing pipeline  
- ML model predicts `Sale_Amount`  
- Outputs downloadable `predictions.csv`  

---

#  **4. Running the Project**

### Create virtual environment:
```
cd marketing_assignment
python -m venv venv
```

### Activate:
```
venv\Scripts\activate
```

### Install:
```
pip install -r requirements.txt
```

### Train model:
```
python -m predictor.train_model
```

### Run Django:
```
cd sale_project
python manage.py migrate
python manage.py runserver
```

Open:
```
http://127.0.0.1:8000/
```

---
