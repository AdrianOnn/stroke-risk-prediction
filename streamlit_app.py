import streamlit as st
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor 
from catboost import CatBoostClassifier, CatBoostRegressor 
import joblib
import pandas as pd
import numpy as np
import urllib.request
import boto3
from botocore.config import Config
from botocore import UNSIGNED
import io

st.set_page_config(
    page_title="Stroke Risk Evaluation", page_icon="🧠", 
    menu_items={
        'Get Help': 'https://www.kaggle.com/code/frankmollard/machine-learning-process-idea-2-app',
        'About': 'Created by Frank Mollard'
    }
)

# Define tabs for the app
tab1, tab2 = st.tabs(["Stroke Risk Assessment", "Model Contributions"])

tab1.header('Stroke Risk Evaluation')
tab1.text(
    """
    Welcome to the Stroke Risk Evaluation app. This tool leverages 
    machine learning algorithms to assess the risk of stroke based on 
    your input data. Please consult a healthcare provider if you suspect a stroke.
    """
)

tab2.title('Model Contributions Breakdown')
tab2.text(
    """
    Explore the percentage contributions of different models to the 
    overall stroke risk assessment.
    """
)

URL="https://strokemodels.s3.eu-central-1.amazonaws.com"

data_load_state1 = tab1.text('Loading models...')
data_load_state2 = tab2.text('Loading models...')

# Function to load models
@st.cache_resource()
def loadAllModels(url):
    models=[]
    for c in ["svm1", "svm2", "logit1", "logit2", "nbc1", "nbc2", "rf1", "rf2", "errGBR"]:
        models.append(
            joblib.load(
                urllib.request.urlopen(url + "/" + "{}.pkl".format(c))
                )
            )
    return models[0], models[1], models[2], models[3], models[4], models[5], models[6], models[7], models[8]

svm1, svm2, logit1, logit2, nbc1, nbc2, rf1, rf2, errGBR = loadAllModels(URL)

# Function to load CatBoost models
@st.cache_resource()
def loadCatBoost(_CB = CatBoostClassifier(), C=["cb1", "cb2"]):
    s3 = boto3.resource(
        service_name='s3',
        region_name='eu-central-1',
        config=Config(signature_version=UNSIGNED)
    )
    bucket = s3.Bucket('strokemodels')
    models=[]
    for c in C:
        obj = bucket.Object("%s" % (c))
        file_stream = io.BytesIO()
        obj.download_fileobj(file_stream)
        models.append(_CB.load_model(blob=file_stream.getvalue()))
    return models

catClassifier = loadCatBoost()
cb1, cb2 = catClassifier[0], catClassifier[0]

catRegressor = loadCatBoost(_CB = CatBoostRegressor(), C=["errCBR"])
errCBR = catRegressor[0]

# Notify that models are loaded
data_load_state1.text("Models Loaded")
data_load_state2.text("Models Loaded")

# Sidebar for input
st.sidebar.title("Patient Information")

work_type = st.sidebar.selectbox('Work Type', ["Child", "Government", "Never worked", "Private", "Self-employed"])
work_type_mapping = {
    "Child": [1, 0, 0, 0],
    "Never worked": [0, 0, 0, 1],
    "Private": [0, 0, 1, 0],
    "Self-employed": [0, 1, 0, 0],
    "Government": [0, 0, 0, 0]
}
work_type_values = work_type_mapping[work_type]
workType = work_type.lower().replace(" ", "_")

age_range = (0, 16) if work_type == "Child" else (17, 100)
age_default = 10 if work_type == "Child" else 40

age = st.sidebar.slider('Age', *age_range, age_default)
bmi = st.sidebar.slider('BMI', 5, 45, 20)
agl = st.sidebar.slider('Average Glucose Level', 50, 400, 100)

smoking_status = st.sidebar.selectbox('Smoking Status', ["Never Smoked", "Formerly Smoked", "Smokes", "Unknown"])
smoking_status_mapping = {
    "Never Smoked": [1, 0, 0],
    "Formerly Smoked": [0, 1, 0],
    "Smokes": [0, 0, 1],
    "Unknown": [0, 0, 0]
}
smoking_status_values = smoking_status_mapping[smoking_status]

heart_disease = 1 if st.sidebar.selectbox('Heart Disease', ["No", "Yes"]) == "Yes" else 0
gender_Male = 1 if st.sidebar.selectbox('Gender', ["Male", "Female"]) == "Male" else 0
ever_married_Yes = 1 if st.sidebar.selectbox('Ever Married', ["Yes", "No"]) == "Yes" else 0
Residence_type_Urban = 1 if st.sidebar.selectbox('Residence Type', ["Urban", "Rural"]) == "Urban" else 0
hypertension = 1 if st.sidebar.selectbox('Hypertension', ["No", "Yes"]) == "Yes" else 0

# Predict risk of stroke
data = pd.DataFrame(
    [[age, hypertension, heart_disease, agl, bmi, gender_Male, *work_type_values, ever_married_Yes, Residence_type_Urban, *smoking_status_values]],
    columns=['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
             'gender_Male', 'work_type_children', 'work_type_Self_employed', 'work_type_Private',
             'work_type_Never_worked', 'ever_married_Yes', 'Residence_type_Urban', 
             'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes']
)

dataC = pd.DataFrame(
    [[age, hypertension, heart_disease, workType, agl, bmi, smoking_status, gender_Male, ever_married_Yes, Residence_type_Urban]],
    columns=['age', 'hypertension', 'heart_disease', 'work_type', 'avg_glucose_level', 'bmi',
             'smoking_status', 'gender_Male', 'ever_married_Yes', 'Residence_type_Urban']
)

contVars = ["age", "avg_glucose_level", "bmi"]

@st.cache_data
def predict(df, dfc, cv: list, weights: list):
    psvm1 = svm1.predict_proba(df[cv])[0][1]
    psvm2 = svm2.predict_proba(df[cv])[0][1]

    pnbc1 = nbc1.predict_proba(df[cv])[0][1]
    pnbc2 = nbc2.predict_proba(df[cv])[0][1]

    prf1 = rf1.predict_proba(df.drop(columns=['work_type_Never_worked']))[0][1]
    prf2 = rf2.predict_proba(df.drop(columns=['work_type_Never_worked']))[0][1]

    plogit1 = logit1.predict_proba(df)[0][1]
    plogit2 = logit2.predict_proba(df)[0][1]
    
    pcb1 = cb1.predict(dfc, prediction_type='Probability')[:, 1]
    pcb2 = cb2.predict(dfc, prediction_type='Probability')[:, 1]

    p = (psvm1 * weights[0] + prf1 * weights[2] + plogit1 * weights[4] + pcb1[0] * weights[6] + pnbc1 * weights[8]) / 2 + \
        (psvm2 * weights[1] + prf2 * weights[3] + plogit2 * weights[5] + pcb2[0] * weights[7] + pnbc2 * weights[9]) / 2

    return p

# Predictions of two Fold Ensembles
pred = predict(data, dataC, contVars, weights=[0.59, 0.11, 0.02, 0.08, 0.13, 0.50, 0.07, 0.26, 0.19, 0.05])

# Error Prediction 
@st.cache_data
def errPred(df):
    er = errCBR.predict(df)[0]
    return er

uncertainty = np.where(errPred(dataC) < 0, 0, errPred(dataC))

# Contributions to the Prediction by Model
@st.cache_data()
def contributions(preds: list):
    c = pd.DataFrame(
        data=preds,
        index=["SVM", "Random Forest", "Logistic Regression", "CatBoost", "Naive Bayes"],
        columns=["Fold 1 Contribution", "Fold 2 Contribution"]
    )
    return c

data_load_state1.text("Prediction done")

# Show the risk of stroke
tab1.metric(
    label="Risk of Stroke", 
    value=f"{round(pred*100/adjst, 3)} %", 
    delta=f"{round(pred*100/adjst-old*100/adjstOld, 4)} percentage points", 
    help="This is the indication for the risk of stroke, given the patient data."
)

tab1.text(f"Confidence in the risk assessment: {round((1-uncertainty)*100,1)} %.")

# Additional Information
def assesBMI(BMI, AGE):
    if BMI > 45 and AGE > 75:
        return """
        Note: Information is unreliable.
        BMI > 45 and age > 75.
        """
    elif BMI <= 10:
        return "BMI level: Too low"
    elif BMI < 18.5:
        return "BMI level: Underweight"
    elif BMI < 25:
        return "BMI level: Normal Weight"
    elif BMI < 30:
        return "BMI level: Overweight"
    elif BMI < 35:
        return "BMI level: Moderate Obesity"
    elif BMI < 40:
        return "BMI level: Severe Obesity"
    else:
        return "BMI level: Extreme Obesity"

tab1.text(assesBMI(bmi, age))

# Data Visualization
viz = dataC.copy()
viz.rename(
    columns={
        "age": "Age",
        "bmi": "BMI",
        "avg_glucose_level": "Average Glucose Level",
        "smoking_status": "Smoking Status",
        "heart_disease": "Heart Disease",
        "gender_Male": "Gender",
        "work_type": "Work Type",
        "ever_married_Yes": "Ever Married",
        "Residence_type_Urban": "Residence Type",
        "hypertension": "Hypertension",    
    }, 
    index={0: 'Data entered'}, 
    inplace=True
)

viz["Hypertension"] = hypertension
viz["Heart Disease"] = heart_disease
viz["Ever Married"] = ever_married_Yes
viz["Work Type"] = work_type
viz["Smoking Status"] = smoking_status
viz["Residence Type"] = Residence_type_Urban
viz["Gender"] = gender_Male

tab1.table(data=viz.T)

# Model Contributions
pred_svm_1 = predict(data, dataC, contVars, weights=[0.59, 0, 0, 0, 0, 0, 0, 0, 0, 0]) * 100/adjst
pred_svm_2 = predict(data, dataC, contVars, weights=[0, 0.11, 0, 0, 0, 0, 0, 0, 0, 0]) * 100/adjst
pred_rf_1 = predict(data, dataC, contVars, weights=[0, 0, 0.02, 0, 0, 0, 0, 0, 0, 0]) * 100/adjst
pred_rf_2 = predict(data, dataC, contVars, weights=[0, 0, 0, 0.08, 0, 0, 0, 0, 0, 0]) * 100/adjst
pred_logit_1 = predict(data, dataC, contVars, weights=[0, 0, 0, 0, 0.13, 0, 0, 0, 0, 0]) * 100/adjst
pred_logit_2 = predict(data, dataC, contVars, weights=[0, 0, 0, 0, 0, 0.50, 0, 0, 0, 0]) * 100/adjst
pred_cb_1 = predict(data, dataC, contVars, weights=[0, 0, 0, 0, 0, 0, 0.07, 0, 0, 0]) * 100/adjst
pred_cb_2 = predict(data, dataC, contVars, weights=[0, 0, 0, 0, 0, 0, 0, 0.26, 0, 0]) * 100/adjst
pred_nbc_1 = predict(data, dataC, contVars, weights=[0, 0, 0, 0, 0, 0, 0, 0, 0.19, 0]) * 100/adjst
pred_nbc_2 = predict(data, dataC, contVars, weights=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0.05]) * 100/adjst

def formater(styler):
    styler.format("{:.4f}")
    styler.background_gradient(cmap="Greens")
    return styler

cont = contributions(
    [
    [pred_svm_1, pred_svm_2],
    [pred_rf_1, pred_rf_2],                
    [pred_logit_1, pred_logit_2],               
    [pred_cb_1, pred_cb_2],                
    [pred_nbc_1, pred_nbc_2],   
    ]
)

tab2.dataframe(
    cont.style.pipe(formater)
)

data_load_state2.text("Prediction done")

tab2.metric(
    label="Risk of Stroke", 
    value=f"{round(pred*100/adjst, 3)} %", 
    help="This is the indication for the risk of stroke, given the patient data."
)
