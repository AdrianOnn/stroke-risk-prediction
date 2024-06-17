import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style
sns.set_theme(style="whitegrid")

# Function to evaluate and plot model performance
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    
    return {
        "conf_matrix": conf_matrix,
        "roc_auc": roc_auc,
        "fpr": fpr,
        "tpr": tpr,
        "accuracy": accuracy,
        "y_pred_proba": y_pred_proba
    }

# Function to preprocess data
@st.cache_data
def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['bmi'] = data['bmi'].fillna(data['bmi'].median())
    data = data[data['gender'].isin(['Male', 'Female'])]
    
    categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    numeric_features = ['age', 'avg_glucose_level', 'bmi']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ])
    
    data_transformed = preprocessor.fit_transform(data.drop(columns=['id']))
    
    # Get feature names after transformation
    numeric_features_transformed = numeric_features
    categorical_features_transformed = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_features_transformed = list(numeric_features_transformed) + list(categorical_features_transformed)
    
    data_transformed = pd.DataFrame(data_transformed, columns=all_features_transformed)
    data_transformed['stroke'] = data['stroke'].values
    
    return data, data_transformed, preprocessor, all_features_transformed

# Function to train and evaluate models with hyperparameter tuning
@st.cache_data
def train_models(X_train, y_train, X_test, y_test):
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_balanced), y=y_train_balanced)
    class_weights_dict = {0: class_weights[0], 1: class_weights[1]}
    
    models = {
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'),
        "XGBoost": XGBClassifier(random_state=42, scale_pos_weight=class_weights_dict[1]/class_weights_dict[0])
    }
    
    param_grids = {
        "Gradient Boosting": {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]},
        "Random Forest": {'n_estimators': [100, 200], 'max_depth': [10, 20]},
        "XGBoost": {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}
    }
    
    tuned_models = {}
    results = {}
    
    for name, model in models.items():
        grid_search = GridSearchCV(model, param_grids[name], cv=3, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train_balanced, y_train_balanced)
        best_model = grid_search.best_estimator_
        tuned_models[name] = best_model
        results[name] = evaluate_model(best_model, X_train_balanced, y_train_balanced, X_test, y_test, name)
    
    return tuned_models, results

# Function to get user inputs
def get_user_inputs(preprocessor, all_features_transformed):
    st.sidebar.header('User Input Features')
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    work_type = st.sidebar.selectbox('Work Type', ('Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'))

    if work_type == 'children':
        age = st.sidebar.slider('Age', 1, 18, 10)
    else:
        age = st.sidebar.slider('Age', 19, 100, 50)

    hypertension = st.sidebar.selectbox('Hypertension', ('No', 'Yes'))
    heart_disease = st.sidebar.selectbox('Heart Disease', ('No', 'Yes'))
    ever_married = st.sidebar.selectbox('Ever Married', ('No', 'Yes'))
    residence_type = st.sidebar.selectbox('Residence Type', ('Urban', 'Rural'))
    avg_glucose_level = st.sidebar.slider('Average Glucose Level', 0, 300, 100)
    bmi = st.sidebar.slider('BMI', 0, 60, 25)
    smoking_status = st.sidebar.selectbox('Smoking Status', ('formerly smoked', 'never smoked', 'smokes', 'Unknown'))
    
    user_data = {
        'gender': gender,
        'age': age,
        'hypertension': 1 if hypertension == 'Yes' else 0,
        'heart_disease': 1 if heart_disease == 'Yes' else 0,
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': residence_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status
    }
    
    user_df = pd.DataFrame(user_data, index=[0])
    user_transformed = preprocessor.transform(user_df)
    user_transformed_df = pd.DataFrame(user_transformed, columns=all_features_transformed)
    
    return user_df, user_transformed_df

# Streamlit app
def main():
    st.set_page_config(
        page_title="Stroke Prediction App",
        page_icon=":hospital:",
        layout="wide"
    )
    
    st.image("cover_image.jpg", use_column_width=False, width=600)
    
    st.title("Stroke Prediction Model")
    
    raw_data, data_transformed, preprocessor, all_features_transformed = preprocess_data('healthcare-dataset-stroke-data.csv')
    
    X = data_transformed.drop('stroke', axis=1)
    y = data_transformed['stroke']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    models, results = train_models(X_train, y_train, X_test, y_test)
    
    user_df, user_transformed_df = get_user_inputs(preprocessor, all_features_transformed)
    
    total_accuracy = sum(result['accuracy'] for result in results.values())
    weights = {name: result['accuracy'] / total_accuracy for name, result in results.items()}
    model_probabilities = np.array([weights[model_name] * models[model_name].predict_proba(user_transformed_df)[:, 1] for model_name in models])
    combined_proba = np.sum(model_probabilities, axis=0)
    combined_percentage = combined_proba[0] * 100

    # Store previous prediction
    if 'prev_combined_percentage' not in st.session_state:
        st.session_state.prev_combined_percentage = combined_percentage
    
    # Calculate percentage difference
    percentage_diff = combined_percentage - st.session_state.prev_combined_percentage
    arrow = "↑" if percentage_diff > 0 else "↓" if percentage_diff < 0 else ""
    percentage_diff_str = f"{arrow} ({abs(percentage_diff):.2f}%)" if percentage_diff != 0 else ""
    
    # Update previous prediction
    st.session_state.prev_combined_percentage = combined_percentage
    
    tab1, tab2, tab3 = st.tabs(["Predict Stroke Risk", "Model Performance Metrics", "Exploratory Data Analysis"])
    
    with tab1:
        st.subheader("User Input Data")
        st.write(user_df)
        
        st.subheader("Model Weights")
        weight_data = [{"Model": model_name, "Weight": f"{weights[model_name]*100:.2f}%"} for model_name in models]
        weight_table = pd.DataFrame(weight_data)
        st.table(weight_table)
        
        st.subheader("Prediction")
        st.markdown(f"<h2 style='color: red; font-weight: bold;'>Combined Probability of Stroke: {combined_percentage:.2f}%</h2>", unsafe_allow_html=True)
        if percentage_diff_str:
            st.markdown(f"<h4 style='color: green;'>{percentage_diff_str}</h4>", unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Model Performance Metrics")
        performance_data = []
        for name, result in results.items():
            performance_data.append({
                "Model": name,
                "Accuracy": result['accuracy'],
                "ROC AUC": result['roc_auc']
            })
        performance_table = pd.DataFrame(performance_data)
        
        # Highlight the best model
        best_model = performance_table.loc[performance_table['Accuracy'].idxmax()]
        st.write(f"**Best Model: {best_model['Model']}**")
        st.table(performance_table.style.apply(lambda x: ['background: lightgreen' if v == best_model['Accuracy'] else '' for v in x], subset=['Accuracy']))
        
        cols = st.columns(2)
        
        for i, (name, result) in enumerate(results.items()):
            with cols[i % 2]:
                st.write(f"### {name}")
                
                fig, ax = plt.subplots(figsize=(5, 3))
                sns.heatmap(result['conf_matrix'], annot=True, fmt='d', cmap='Blues', xticklabels=['No Stroke', 'Stroke'], yticklabels=['No Stroke', 'Stroke'], ax=ax)
                ax.set_title(f'Confusion Matrix - {name}')
                ax.set_xlabel('Predicted Label')
                ax.set_ylabel('True Label')
                st.pyplot(fig)
                
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.plot(result['fpr'], result['tpr'], label=f'ROC curve (area = {result["roc_auc"]:.2f})')
                ax.plot([0, 1], [0, 1], 'k--')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title(f'ROC Curve - {name}')
                ax.legend(loc='lower right')
                st.pyplot(fig)

    with tab3:
        st.subheader("Exploratory Data Analysis")
        
        st.write("### Distribution of Numeric Features")
        num_cols = st.columns(3)
        for i, col in enumerate(['age', 'avg_glucose_level', 'bmi']):
            with num_cols[i]:
                fig, ax = plt.subplots(figsize=(5, 3))
                sns.histplot(raw_data[col], kde=True, ax=ax)
                ax.set_title(f'Distribution of {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                st.pyplot(fig)
        
        st.write("### Counts of Categorical Features")
        cat_cols = st.columns(3)
        for i, col in enumerate(['gender', 'ever_married', 'work_type']):
            with cat_cols[i]:
                fig, ax = plt.subplots(figsize=(5, 3))
                sns.countplot(x=col, data=raw_data, ax=ax)
                ax.set_title(f'Count of {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Count')
                if col == 'work_type':
                    plt.xticks(rotation=45)
                st.pyplot(fig)
        
        more_cat_cols = st.columns(2)
        for i, col in enumerate(['Residence_type', 'smoking_status']):
            with more_cat_cols[i]:
                fig, ax = plt.subplots(figsize=(5, 3))
                sns.countplot(x=col, data=raw_data, ax=ax)
                ax.set_title(f'Count of {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Count')
                if col == 'smoking_status':
                    plt.xticks(rotation=45)
                st.pyplot(fig)
        
        st.write("### Correlation Heatmap")
        encoded_data = pd.get_dummies(raw_data.drop(columns=['id']), drop_first=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = encoded_data.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, annot=True, mask=mask, cmap='coolwarm', ax=ax, fmt='.2f', annot_kws={"size": 8}, cbar_kws={"shrink": 0.75})
        ax.set_title('Correlation Heatmap')
        st.pyplot(fig)

if __name__ == '__main__':
    main()
