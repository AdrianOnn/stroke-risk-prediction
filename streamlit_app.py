from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix, accuracy_score
import pandas as pd
import math
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
import streamlit as st
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid")
pd.set_option('future.no_silent_downcasting', True)

@st.cache_data
def data_cleaning(df):
    """
    Cleans the dataset by renaming columns, replacing values, and removing unnecessary columns.
    Args:
        df (pd.DataFrame): The input dataframe.
    Returns:
        pd.DataFrame: The cleaned dataframe.
    """
    df = df.replace({'work_type': {'Private': 'Private Industry', 'Govt_job': 'Government Job', 'children': 'Not of Working Age', 'Never_worked': 'Not of Working Age'}})
    df = df.replace({'smoking_status': {'formerly smoked': 'Former Smoker', 'never smoked': 'Never Smoked', 'smokes': 'Smoker'}})
    df = df.rename(columns={'gender': 'Is_Male', 'Residence_type': 'Is_Urban_Residence'})
    df = df.loc[df['Is_Male'] != 'Other']
    df['age'] = df['age'].apply(lambda x: math.ceil(x))
    df = df.replace({'Is_Male': {'Male': 1, 'Female': 0}, 'Is_Urban_Residence': {'Urban': 1, 'Rural': 0}, 'ever_married': {'Yes': 1, 'No': 0}})
    df = df.astype({'stroke': 'bool'})
    if 'id' in df.columns:
        df = df.drop(['id'], axis=1)
    return df

@st.cache_data
def data_engrg(df):
    """
    Fills missing values, transforms categorical variables using one-hot encoding, and prepares the target variable.
    Args:
        df (pd.DataFrame): The cleaned dataframe.
    Returns:
        pd.DataFrame: The dataframe with engineered features.
    """
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    df_target = df[['stroke']]
    df = df.drop(['stroke'], axis=1)
    df = pd.get_dummies(df, columns=['work_type', 'smoking_status'])
    df = df.join(df_target)
    return df

@st.cache_data
def preprocess_data(file_path):
    """
    Reads the CSV file and performs data cleaning and feature engineering.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        tuple: A tuple containing the raw and processed dataframes.
    """
    df = pd.read_csv(file_path)
    df_cleaned = data_cleaning(df)
    df_engineered = data_engrg(df_cleaned)
    return df, df_engineered

def predictor_trainer(predictor, x_test, x_train, y_train):
    """
    Trains the given predictor and returns the trained model along with predictions and probabilities.
    Args:
        predictor (model): The machine learning model to train.
        x_test (pd.DataFrame): Test features.
        x_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
    Returns:
        tuple: A tuple containing the trained model, predictions, and prediction probabilities.
    """
    predictor.fit(x_train, y_train)
    y_pred = predictor.predict(x_test)
    y_proba = predictor.predict_proba(x_test)[:, 1]
    return predictor, y_pred, y_proba

def generate_roc(x_test, y_test, predlabel, y_proba):
    """
    Generates the ROC curve and calculates AUC for the given model.
    Args:
        x_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.
        predlabel (str): Label for the predictor.
        y_proba (np.array): Prediction probabilities.
    Returns:
        tuple: A tuple containing the AUC score and the ROC curve figure.
    """
    ns_probs = [0 for _ in range(len(y_test))]
    lr_auc = roc_auc_score(y_test, y_proba)
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(ns_fpr, ns_tpr, linestyle='--')
    ax.plot(lr_fpr, lr_tpr, marker='.', label='%s (AUC = %.3f)' % (predlabel, lr_auc))
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.legend(loc="lower right")
    ax.set_title(f'{predlabel} ROC Curve')
    return lr_auc, fig

@st.cache_resource
def train_models(x_train, y_train, x_test, y_test):
    """
    Trains multiple models (Logistic Regression, Random Forest, K-Nearest Neighbors) and evaluates their performance.
    Args:
        x_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        x_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.
    Returns:
        dict: A dictionary containing performance metrics and plots for each model.
    """
    results = {}

    # Logistic Regression
    predlabel_lr = 'Logistic Regression'
    param_dist_lr = {"C": [0.1, 0.5, 1, 5, 10], "solver": ['lbfgs', 'liblinear']}
    predictor_lr = LogisticRegression(max_iter=2000)
    predictor_cv_lr = RandomizedSearchCV(predictor_lr, param_dist_lr, cv=5)
    predictor_lr, y_pred_lr, y_proba_lr = predictor_trainer(predictor_cv_lr, x_test, x_train, y_train)
    lr_auc, plt_lr = generate_roc(x_test, y_test, predlabel_lr, y_proba_lr)
    lr_conf_matrix = confusion_matrix(y_test, y_pred_lr)
    lr_class_report = classification_report(y_test, y_pred_lr, output_dict=True)
    lr_accuracy = accuracy_score(y_test, y_pred_lr)

    results['Logistic Regression'] = {
        'auc': lr_auc,
        'plot': plt_lr,
        'conf_matrix': lr_conf_matrix,
        'class_report': lr_class_report,
        'accuracy': lr_accuracy,
        'y_proba': y_proba_lr
    }

    # Random Forest
    predlabel_rf = "Random Forest"
    param_dist_rf = {"n_estimators": randint(10, 150), "max_depth": [None, 1, 2, 3, 4, 5, 6, 7]}
    predictor_rf = RandomForestClassifier()
    predictor_cv_rf = RandomizedSearchCV(predictor_rf, param_dist_rf, cv=5)
    predictor_rf, y_pred_rf, y_proba_rf = predictor_trainer(predictor_cv_rf, x_test, x_train, y_train)
    rf_auc, plt_rf = generate_roc(x_test, y_test, predlabel_rf, y_proba_rf)
    rf_conf_matrix = confusion_matrix(y_test, y_pred_rf)
    rf_class_report = classification_report(y_test, y_pred_rf, output_dict=True)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)

    results['Random Forest'] = {
        'auc': rf_auc,
        'plot': plt_rf,
        'conf_matrix': rf_conf_matrix,
        'class_report': rf_class_report,
        'accuracy': rf_accuracy,
        'y_proba': y_proba_rf
    }

    # K-Nearest Neighbours
    predlabel_knn = 'K-Nearest Neighbours'
    param_dist_knn = {"n_neighbors": randint(1, 6)}
    predictor_knn = KNeighborsClassifier()
    predictor_cv_knn = RandomizedSearchCV(predictor_knn, param_dist_knn, cv=5)
    predictor_knn, y_pred_knn, y_proba_knn = predictor_trainer(predictor_cv_knn, x_test, x_train, y_train)
    knn_auc, plt_knn = generate_roc(x_test, y_test, predlabel_knn, y_proba_knn)
    knn_conf_matrix = confusion_matrix(y_test, y_pred_knn)
    knn_class_report = classification_report(y_test, y_pred_knn, output_dict=True)
    knn_accuracy = accuracy_score(y_test, y_pred_knn)

    results['K-Nearest Neighbours'] = {
        'auc': knn_auc,
        'plot': plt_knn,
        'conf_matrix': knn_conf_matrix,
        'class_report': knn_class_report,
        'accuracy': knn_accuracy,
        'y_proba': y_proba_knn
    }
    
    return results

def get_user_inputs():
    """
    Collects user input parameters from the Streamlit sidebar.
    Returns:
        tuple: A tuple containing the user input dataframe and the transformed dataframe.
    """
    st.sidebar.header('User Input Parameters')

    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    work_type = st.sidebar.selectbox('Work Type', ['Private Industry', 'Government Job', 'Self-employed', 'Not of Working Age'])
    
    if work_type == 'Not of Working Age':
        age = st.sidebar.slider('Age', 0, 18, 10)
    else:
        age = st.sidebar.slider('Age', 18, 100, 25)
    
    hypertension = st.sidebar.selectbox('Hypertension', ['No', 'Yes'])
    heart_disease = st.sidebar.selectbox('Heart Disease', ['No', 'Yes'])
    ever_married = st.sidebar.selectbox('Ever Married', ['No', 'Yes'])
    residence_type = st.sidebar.selectbox('Residence Type', ['Rural', 'Urban'])
    avg_glucose_level = st.sidebar.slider('Average Glucose Level', 0.0, 300.0, 100.0)
    bmi = st.sidebar.slider('BMI', 0.0, 60.0, 25.0)
    smoking_status = st.sidebar.selectbox('Smoking Status', ['Never Smoked', 'Former Smoker', 'Smoker'])
    
    user_data = {
        'Is_Male': 1 if gender == 'Male' else 0,
        'age': age,
        'hypertension': 1 if hypertension == 'Yes' else 0,
        'heart_disease': 1 if heart_disease == 'Yes' else 0,
        'ever_married': 1 if ever_married == 'Yes' else 0,
        'Is_Urban_Residence': 1 if residence_type == 'Urban' else 0,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status,
        'work_type': work_type
    }
    
    user_df = pd.DataFrame(user_data, index=[0])
    user_transformed_df = pd.get_dummies(user_df, columns=['work_type', 'smoking_status'])

    # Ensure all required columns are present
    all_columns = ['Is_Male', 'age', 'hypertension', 'heart_disease', 'ever_married', 'Is_Urban_Residence', 'avg_glucose_level', 'bmi',
                   'work_type_Government Job', 'work_type_Not of Working Age', 'work_type_Private Industry', 'work_type_Self-employed',
                   'smoking_status_Former Smoker', 'smoking_status_Never Smoked', 'smoking_status_Smoker', 'smoking_status_Unknown']
    for col in all_columns:
        if col not in user_transformed_df.columns:
            user_transformed_df[col] = 0

    user_transformed_df = user_transformed_df[all_columns]
    
    return user_df, user_transformed_df

def main():
    """
    Main function to run the Streamlit app. Sets up the page configuration, loads data, trains models, and displays the results.
    """
    st.set_page_config(
        page_title="Stroke Prediction App",
        page_icon=":hospital:",
        layout="wide"
    )
    
    st.image("cover_image.jpg", use_column_width=False, width=600)
    
    st.title("Stroke Prediction Model")
    
    file_path = 'healthcare-dataset-stroke-data.csv'
    raw_data, data_transformed = preprocess_data(file_path)
    
    x = data_transformed.iloc[:, :-1]
    y = data_transformed.iloc[:, -1]
    smote = SMOTE(random_state=42)
    x_smote, y_smote = smote.fit_resample(x, y)
    x_train, x_test, y_train, y_test = train_test_split(x_smote, y_smote, test_size=0.2, random_state=23, stratify=y_smote)
    
    results = train_models(x_train, y_train, x_test, y_test)
    
    tab1, tab2, tab3 = st.tabs(["Predict Stroke Risk", "Model Performance Metrics", "Best Model"])
    
    with tab1:
        st.subheader("User Input Data")
        user_df, user_transformed_df = get_user_inputs()
        st.write(user_df)

        # Predicting risk for the new user input
        if 'previous_risk' not in st.session_state:
            st.session_state['previous_risk'] = None
        
        best_model_name = max(results, key=lambda k: results[k]['auc'])
        best_model_instance = results[best_model_name]

        if best_model_name == 'Random Forest':
            model = RandomForestClassifier()
        elif best_model_name == 'Logistic Regression':
            model = LogisticRegression(max_iter=2000)
        elif best_model_name == 'K-Nearest Neighbours':
            model = KNeighborsClassifier()

        model.fit(x_train, y_train)
        user_risk = model.predict_proba(user_transformed_df)[:, 1][0] * 100

        if st.session_state['previous_risk'] is not None:
            risk_change = user_risk - st.session_state['previous_risk']
            if risk_change != 0:
                risk_change_str = f"{abs(risk_change):.2f}% {'increase' if risk_change > 0 else 'decrease'}"
                risk_change_arrow = "↑" if risk_change > 0 else "↓"
                risk_change_color = "red" if risk_change > 0 else "green"

                st.markdown(
                    f"<h2 style='color: yellow; font-weight: bold;'>Predicted Stroke Risk: {user_risk:.2f}%</h2>"
                    f"<h3 style='color: {risk_change_color};'>{risk_change_arrow} {risk_change_str}</h3>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<h2 style='color: yellow; font-weight: bold;'>Predicted Stroke Risk: {user_risk:.2f}%</h2>",
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                f"<h2 style='color: yellow; font-weight: bold;'>Predicted Stroke Risk: {user_risk:.2f}%</h2>",
                unsafe_allow_html=True
            )
        
        st.session_state['previous_risk'] = user_risk
    
    with tab2:
        st.subheader("Model Performance Metrics")
        metrics_data = []
        for model_name, result in results.items():
            report = result['class_report']
            metrics_data.append({
                "Model": model_name,
                "Precision": report['1']['precision'] if '1' in report else report['True']['precision'],
                "Recall": report['1']['recall'] if '1' in report else report['True']['recall'],
                "F1-Score": report['1']['f1-score'] if '1' in report else report['True']['f1-score'],
                "Accuracy": result['accuracy'],
                "AUC": result['auc']
            })
        metrics_df = pd.DataFrame(metrics_data)
        st.table(metrics_df)

        st.subheader("Confusion Matrices and ROC Curves")
        col1, col2 = st.columns(2)
        for model_name, result in results.items():
            with col1:
                st.write(f"### {model_name} Confusion Matrix")
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.heatmap(result['conf_matrix'], annot=True, fmt='d', cmap='Blues', xticklabels=['No Stroke', 'Stroke'], yticklabels=['No Stroke', 'Stroke'], ax=ax)
                ax.set_title(f'{model_name} Confusion Matrix')
                ax.set_xlabel('Predicted Label')
                ax.set_ylabel('True Label')
                st.pyplot(fig)
            with col2:
                st.write(f"### {model_name} ROC Curve")
                st.pyplot(result['plot'])

    with tab3:
        st.subheader("Best Model Performance Metrics and ROC Curve")
        best_model_name = max(results, key=lambda k: results[k]['auc'])
        best_model_result = results[best_model_name]
        
        st.write(f"### Best Model: {best_model_name}")
        best_metrics = {
            "Precision": best_model_result['class_report']['1']['precision'] if '1' in best_model_result['class_report'] else best_model_result['class_report']['True']['precision'],
            "Recall": best_model_result['class_report']['1']['recall'] if '1' in best_model_result['class_report'] else best_model_result['class_report']['True']['recall'],
            "F1-Score": best_model_result['class_report']['1']['f1-score'] if '1' in best_model_result['class_report'] else best_model_result['class_report']['True']['f1-score'],
            "Accuracy": best_model_result['accuracy'],
            "AUC": best_model_result['auc']
        }
        st.table(pd.DataFrame([best_metrics]))

        st.write(f"### {best_model_name} Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(best_model_result['conf_matrix'], annot=True, fmt='d', cmap='Blues', xticklabels=['No Stroke', 'Stroke'], yticklabels=['No Stroke', 'Stroke'], ax=ax)
        ax.set_title(f'{best_model_name} Confusion Matrix')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        st.pyplot(fig)

        st.write(f"### {best_model_name} ROC Curve")
        fig, ax = plt.subplots(figsize=(5, 3))
        for model_name, result in results.items():
            lr_probs = result['y_proba']
            lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
            ax.plot(lr_fpr, lr_tpr, marker='.', label=f'{model_name} (AUC = {result["auc"]:.3f})')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Final ROC Curves for All Classifiers')
        ax.legend(loc='lower right')
        st.pyplot(fig)

if __name__ == '__main__':
    main()
