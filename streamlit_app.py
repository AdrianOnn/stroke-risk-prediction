import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
@st.cache
def load_data():
    # Load the dataset from Kaggle
    url = "https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset"
    # Replace 'path/to/your/kaggle/credentials' with the actual path to your Kaggle credentials file
    # You need to download your Kaggle API key from your Kaggle account settings and save it locally
    kaggle_credentials_path = "C:\Users\User\Desktop\kaggle.json"
    # Specify the name of the dataset file
    dataset_file_name = "healthcare-dataset-stroke-data.csv"
    # Load the dataset into a DataFrame
    dataset = pd.read_csv(f'{url}/{dataset_file_name}')
    return dataset

# Preprocess the data
def preprocess_data(dataset):
    # Fill missing values
    dataset['bmi'].fillna(dataset['bmi'].median(), inplace=True)
    # Convert categorical variables to numerical using one-hot encoding
    dataset = pd.get_dummies(dataset, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])
    return dataset

# Train the model
def train_models(dataset):
    # Split the dataset into features and target variable
    X = dataset.drop(columns=['id', 'stroke'])
    y = dataset['stroke']
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Initialize the classifiers
    rf_model = RandomForestClassifier()
    logit_model = LogisticRegression()
    svm_model = SVC()
    # Train the models
    rf_model.fit(X_train, y_train)
    logit_model.fit(X_train, y_train)
    svm_model.fit(X_train, y_train)
    return rf_model, logit_model, svm_model

# Predict stroke
def predict_stroke(data, models):
    rf_model, logit_model, svm_model = models
    rf_prediction = rf_model.predict(data)[0]
    logit_prediction = logit_model.predict(data)[0]
    svm_prediction = svm_model.predict(data)[0]
    return rf_prediction, logit_prediction, svm_prediction

# Load the dataset
dataset = load_data()
# Preprocess the data
dataset = preprocess_data(dataset)
# Train the models
rf_model, logit_model, svm_model = train_models(dataset)

# Display the form for user input
st.title("Stroke Prediction")
st.write("Please enter your details below:")

age = st.slider("Age", min_value=0, max_value=100, value=50)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
avg_glucose_level = st.slider("Average Glucose Level", min_value=50, max_value=400, value=150)
bmi = st.slider("BMI", min_value=10, max_value=50, value=25)
gender = st.selectbox("Gender", ["Male", "Female"])
ever_married = st.selectbox("Ever Married", ["Yes", "No"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

gender_Male = 1 if gender == "Male" else 0
ever_married_Yes = 1 if ever_married == "Yes" else 0
work_type_columns = [f"work_type_{work_type}" if col.endswith(work_type) else col for col in dataset.columns]
residence_type_Urban = 1 if residence_type == "Urban" else 0
smoking_status_columns = [f"smoking_status_{smoking_status}" if col.endswith(smoking_status) else col for col in dataset.columns]

input_data = {
    'age': age,
    'hypertension': hypertension,
    'heart_disease': heart_disease,
    'avg_glucose_level': avg_glucose_level,
    'bmi': bmi,
    'gender_Male': gender_Male,
    'ever_married_Yes': ever_married_Yes,
    'work_type_columns': work_type_columns,
    'residence_type_Urban': residence_type_Urban,
    'smoking_status_columns': smoking_status_columns
}

# Predict stroke
models = (rf_model, logit_model, svm_model)
rf_prediction, logit_prediction, svm_prediction = predict_stroke(pd.DataFrame([input_data]), models)

# Display the prediction
st.write("Prediction:")
st.write(f"Random Forest: {'Yes' if rf_prediction else 'No'}")
st.write(f"Logistic Regression: {'Yes' if logit_prediction else 'No'}")
st.write(f"SVM: {'Yes' if svm_prediction else 'No'}")
