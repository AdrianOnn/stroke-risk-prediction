import streamlit as st
import pandas as pd
import numpy as np

# Sample prediction function (replace with your actual model)
def predict(data, weights):
    """
    Simulate a prediction based on input data and weights.
    This is a dummy function for demonstration purposes.
    """
    # Assume the input data has been preprocessed and normalized
    data_values = data.values.flatten()
    prediction = np.dot(data_values, weights)
    return prediction

# Simulated data and variables
data = pd.DataFrame({'age': [65], 'hypertension': [1], 'heart_disease': [0], 'ever_married': [1], 'work_type': ['Private'], 'Residence_type': ['Urban'], 'avg_glucose_level': [105], 'bmi': [28], 'smoking_status': ['formerly smoked'], 'gender_Male': [1]})
dataC = data.copy()
contVars = data.columns.tolist()
age = data['age'].values[0]
bmi = data['bmi'].values[0]
adjst = 1
adjstOld = 1
old = 0.5
uncertainty = 0.1

st.title("Stroke Risk Prediction")

tab1, tab2 = st.tabs(["Patient Data", "Model Contributions"])

with tab1:
    st.header("Patient Data")
    
    # Show the risk of stroke
    weights = np.random.random(data.shape[1])  # Random weights for demonstration
    pred = predict(data, weights)
    st.metric(label="Risk of Stroke", value=f"{round(pred*100/adjst, 3)} %", delta=f"{round(pred*100/adjst-old*100/adjstOld, 4)} percentage points", help="This is the indication for the risk of stroke, given the patient data.")
    st.text(f"Confidence in the risk assessment: {round((1-uncertainty)*100,1)} %.")
    
    # Additional Information
    def assesBMI(BMI, AGE):
        if BMI > 45 and AGE > 75:
            return "Note: Information is unreliable. BMI > 45 and age > 75."
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
    
    st.text(assesBMI(bmi, age))
    
    # Data Visualization
    viz = data.copy()
    viz.rename(
        columns={
            "age": "Age",
            "bmi": "BMI",
            "avg_glucose_level": "Average Glucose Level",
            "smoking_status": "Smoking Status",
            "heart_disease": "Heart Disease",
            "gender_Male": "Gender",
            "work_type": "Work Type",
            "ever_married": "Ever Married",
            "Residence_type": "Residence Type",
            "hypertension": "Hypertension",    
        }, 
        index={0: 'Data entered'}, 
        inplace=True
    )
    
    st.table(data=viz.T)

with tab2:
    st.header("Model Contributions")
    
    # Model Contributions
    pred_svm_1 = predict(data, weights=[0.59, 0, 0, 0, 0, 0, 0, 0, 0, 0]) * 100/adjst
    pred_svm_2 = predict(data, weights=[0, 0.11, 0, 0, 0, 0, 0, 0, 0, 0]) * 100/adjst
    pred_rf_1 = predict(data, weights=[0, 0, 0.02, 0, 0, 0, 0, 0, 0, 0]) * 100/adjst
    pred_rf_2 = predict(data, weights=[0, 0, 0, 0.08, 0, 0, 0, 0, 0, 0]) * 100/adjst
    pred_logit_1 = predict(data, weights=[0, 0, 0, 0, 0.13, 0, 0, 0, 0, 0]) * 100/adjst
    pred_logit_2 = predict(data, weights=[0, 0, 0, 0, 0, 0.50, 0, 0, 0, 0]) * 100/adjst
    pred_cb_1 = predict(data, weights=[0, 0, 0, 0, 0, 0, 0.07, 0, 0, 0]) * 100/adjst
    pred_cb_2 = predict(data, weights=[0, 0, 0, 0, 0, 0, 0, 0.26, 0, 0]) * 100/adjst
    pred_nbc_1 = predict(data, weights=[0, 0, 0, 0, 0, 0, 0, 0, 0.19, 0]) * 100/adjst
    pred_nbc_2 = predict(data, weights=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0.05]) * 100/adjst
    
    def contributions(contributions):
        c = pd.DataFrame(
            contributions, 
            columns=["Fold 1 Contribution", "Fold 2 Contribution"], 
            index=[
                "SVM Contribution", 
                "Random Forest Contribution", 
                "Logistic Regression Contribution", 
                "CatBoost Contribution", 
                "Naive Bayes Contribution"
            ]
        )
        return c
    
    cont = contributions(
        [
            [pred_svm_1, pred_svm_2],
            [pred_rf_1, pred_rf_2],                
            [pred_logit_1, pred_logit_2],               
            [pred_cb_1, pred_cb_2],                
            [pred_nbc_1, pred_nbc_2],   
        ]
    )
    
    def formater(styler):
        styler.format("{:.4f}")
        styler.background_gradient(cmap="Greens")
        return styler
    
    st.dataframe(cont.style.pipe(formater))
    
    st.metric(label="Risk of Stroke", value=f"{round(pred*100/adjst, 3)} %", help="This is the indication for the risk of stroke, given the patient data.")
