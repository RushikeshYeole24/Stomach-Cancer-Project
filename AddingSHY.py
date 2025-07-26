import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Load datasets
stage_data = pd.read_csv('stagepred.csv')
survival_data = pd.read_csv('updated.csv')
treatment_response = pd.read_csv('treatmentresponse.csv')
treatment_chemo = pd.read_csv('Chemo_prediction.csv')


@st.cache(allow_output_mutation=True)
def preprocess_data(data, selected_features, target_variable):
    # Drop rows with missing values
    data = data.dropna(subset=selected_features + [target_variable])

    # Separate features and target variable
    X = data[selected_features]
    y = data[target_variable]

    # Convert categorical variables to dummy variables
    X = pd.get_dummies(X)

    return X, y

@st.cache(allow_output_mutation=True)
def train_decision_tree(X_train, y_train):
    # Initialize and train the decision tree model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

@st.cache(allow_output_mutation=True)
def train_logistic_regression(X_train, y_train):
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Initialize and train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    return model, scaler

@st.cache(allow_output_mutation=True)
def train_random_forest(X_train, y_train):
    # Initialize and train the Random Forest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model, None  # Return None for scaler since Random Forest doesn't require scaling

def generate_shap_explanation(model, X_train, input_data, model_type='tree', scaler=None):
    """Generate SHAP explanations for a prediction"""
    try:
        # Initialize SHAP explainer based on model type
        if model_type == 'tree':
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_data)
        else:  # linear model
            if scaler:
                X_train_scaled = scaler.transform(X_train)
                explainer = shap.LinearExplainer(model, X_train_scaled)
                shap_values = explainer.shap_values(input_data)
            else:
                explainer = shap.LinearExplainer(model, X_train)
                shap_values = explainer.shap_values(input_data)
        
        return explainer, shap_values
    except Exception as e:
        st.error(f"Error generating SHAP explanation: {str(e)}")
        return None, None

def display_shap_plots(explainer, shap_values, input_data, feature_names):
    """Display SHAP plots in Streamlit"""
    try:
        st.subheader("SHAP Explanation")
        
        # For multi-class classification, handle different output formats
        if isinstance(shap_values, list):
            # Multi-class case - use the first class for visualization
            shap_vals_to_plot = shap_values[0] if len(shap_values) > 0 else shap_values
        else:
            shap_vals_to_plot = shap_values
        
        # Waterfall plot
        st.subheader("Feature Impact on Prediction (Waterfall Plot)")
        fig_waterfall = plt.figure(figsize=(10, 6))
        if hasattr(shap, 'plots') and hasattr(shap.plots, 'waterfall'):
            # For newer SHAP versions
            shap.plots.waterfall(explainer.expected_value[0] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value, 
                               shap_vals_to_plot[0] if len(shap_vals_to_plot.shape) > 1 else shap_vals_to_plot, 
                               input_data.iloc[0] if hasattr(input_data, 'iloc') else input_data[0])
        else:
            # For older SHAP versions
            shap.waterfall_plot(explainer.expected_value[0] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value, 
                              shap_vals_to_plot[0] if len(shap_vals_to_plot.shape) > 1 else shap_vals_to_plot, 
                              input_data.iloc[0] if hasattr(input_data, 'iloc') else input_data[0])
        st.pyplot(fig_waterfall)
        plt.close()
        
        # Feature importance bar plot
        st.subheader("Feature Importance")
        if len(shap_vals_to_plot.shape) > 1:
            feature_importance = np.abs(shap_vals_to_plot[0])
        else:
            feature_importance = np.abs(shap_vals_to_plot)
        
        # Create a DataFrame for better visualization
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=True)
        
        fig_bar = px.bar(importance_df.tail(10), x='Importance', y='Feature', 
                        orientation='h', title='Top 10 Most Important Features')
        st.plotly_chart(fig_bar)
        
        # Summary statistics
        st.subheader("SHAP Values Summary")
        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP Value': shap_vals_to_plot[0] if len(shap_vals_to_plot.shape) > 1 else shap_vals_to_plot,
            'Feature Value': input_data.iloc[0] if hasattr(input_data, 'iloc') else input_data[0]
        })
        st.dataframe(shap_df.sort_values('SHAP Value', key=abs, ascending=False))
        
    except Exception as e:
        st.error(f"Error displaying SHAP plots: {str(e)}")
        # Fallback: simple feature importance
        try:
            if len(shap_vals_to_plot.shape) > 1:
                importance = np.abs(shap_vals_to_plot[0])
            else:
                importance = np.abs(shap_vals_to_plot)
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            st.subheader("Feature Importance (Simplified)")
            st.dataframe(importance_df.head(10))
        except:
            st.error("Unable to display SHAP explanations")

# Sidebar navigation
page = st.sidebar.selectbox("Navigate to", ["Home", "Stage Prediction", "Survival Prediction", "Treatment Response","Chemotherapy Prediction"])

if page == "Home":
    st.subheader('Welcome to Cancer Prediction System')
    st.write('This system helps predict cancer stages and survival status based on patient data.')

elif page == "Stage Prediction":
    st.title('Cancer Stage Prediction')

    selected_features_stage = [
        "Age recode with <1 year olds and 90+",
        "Sex",
        "Year of diagnosis",
        "Race recode (White, Black, Other)",
        "TNM 7/CS v0204+ Schema (thru 2017)",
        "Diagnostic Confirmation",
        "CS tumor size (2004-2015)",
        "CS extension (2004-2015)",
        "CS lymph nodes (2004-2015)",
        "CS mets at dx (2004-2015)"
    ]
    target_variable_stage = 'Combined Summary Stage (2004+)'

    X_stage, y_stage = preprocess_data(stage_data, selected_features_stage, target_variable_stage)
    X_train_stage, X_test_stage, y_train_stage, y_test_stage = train_test_split(X_stage, y_stage, test_size=0.2, random_state=42)

    model_stage, _ = train_random_forest(X_train_stage, y_train_stage)

    # User input form for stage prediction
    st.subheader('Enter Patient Data for Stage Prediction')
    input_data_stage = {}
    for feature in selected_features_stage:
        input_data_stage[feature] = st.text_input(feature)

    if st.button('Predict Stage'):
        input_df_stage = pd.DataFrame([input_data_stage], columns=selected_features_stage)
        input_df_stage = pd.get_dummies(input_df_stage)
        missing_cols_stage = set(X_train_stage.columns) - set(input_df_stage.columns)
        for col in missing_cols_stage:
            input_df_stage[col] = 0
        input_df_stage = input_df_stage[X_train_stage.columns]
        prediction_stage = model_stage.predict(input_df_stage)
        st.success(f'Predicted stage: {prediction_stage[0]}')
        
        # Generate SHAP explanation
        explainer_stage, shap_values_stage = generate_shap_explanation(
            model_stage, X_train_stage, input_df_stage, model_type='tree'
        )
        
        if explainer_stage is not None and shap_values_stage is not None:
            display_shap_plots(explainer_stage, shap_values_stage, input_df_stage, X_train_stage.columns)

elif page == "Survival Prediction":
    st.title('Cancer Survival Prediction')

    selected_features_survival = [
        'Derived EOD 2018 T (2018+)',
        'Derived EOD 2018 N (2018+)',
        'Derived EOD 2018 M (2018+)',
        'Months from diagnosis to treatment',
        'TNM 7/CS v0204+ Schema recode',
        'EOD Schema ID Recode (2010+)',
        'Site recode - rare tumors',
        'Derived EOD 2018 Stage Group (2018+)',
        'RX Summ--Scope Reg LN Sur (2003+)',
        'EOD Primary Tumor (2018+)',
        'EOD Regional Nodes (2018+)',
        'EOD Mets (2018+)',
        'Tumor Size Summary (2016+)',
        'Regional nodes positive (1988+)',
    ]
    target_variable_survival = 'Vital status recode (study cutoff used)'

    X_survival, y_survival = preprocess_data(survival_data, selected_features_survival, target_variable_survival)
    X_train_survival, X_test_survival, y_train_survival, y_test_survival = train_test_split(X_survival, y_survival, test_size=0.2, random_state=42)

    model_survival, scaler_survival = train_logistic_regression(X_train_survival, y_train_survival)

    # User input form for survival prediction
    st.subheader('Enter Patient Data for Survival Prediction')
    input_data_survival = {}
    for feature in selected_features_survival:
        input_data_survival[feature] = st.text_input(feature)

    if st.button('Predict Survival'):
        input_df_survival = pd.DataFrame([input_data_survival], columns=selected_features_survival)
        input_df_survival = pd.get_dummies(input_df_survival)
        missing_cols_survival = set(X_train_survival.columns) - set(input_df_survival.columns)
        for col in missing_cols_survival:
            input_df_survival[col] = 0
        input_df_survival = input_df_survival[X_train_survival.columns]
        input_scaled_survival = scaler_survival.transform(input_df_survival)
        prediction_survival = model_survival.predict(input_scaled_survival)
        st.success(f'Predicted survival status: {prediction_survival[0]}')
        
        # Generate SHAP explanation
        explainer_survival, shap_values_survival = generate_shap_explanation(
            model_survival, X_train_survival, input_scaled_survival, model_type='linear', scaler=scaler_survival
        )
        
        if explainer_survival is not None and shap_values_survival is not None:
            display_shap_plots(explainer_survival, shap_values_survival, input_df_survival, X_train_survival.columns)
        
elif page == "Treatment Response":
    st.title('Treatment Response Prediction')

    selected_features_treatment =  [
        "Age recode with <1 year olds",
        "Sex",
        "Year of diagnosis",
        "Race recode (W, B, AI, API)",
        "Site recode ICD-O-3/WHO 2008",
        "TNM 7/CS v0204+ Schema (thru 2017)",
        "Chemotherapy recode (yes, no/unk)",
        "Radiation recode",
        "RX Summ--Systemic/Sur Seq (2007+)",
        "Months from diagnosis to treatment",
        "Regional nodes examined (1988+)",
        "Regional nodes positive (1988+)",
        "CS tumor size (2004-2015)",
        "CS extension (2004-2015)",
        "CS lymph nodes (2004-2015)",
        "CS mets at dx (2004-2015)",
        "Survival months"
    ]
    target_variable_treatment = "RX Summ--Surg/Rad Seq"  # Corrected target variable
    
    X_treatment, y_treatment = preprocess_data(treatment_response, selected_features_treatment, target_variable_treatment)
    X_train_treatment, X_test_treatment, y_train_treatment, y_test_treatment = train_test_split(X_treatment, y_treatment, test_size=0.2, random_state=42)
    
    model_treatment, _ = train_random_forest(X_train_treatment, y_train_treatment)

    # User input form for treatment response prediction
    st.subheader('Enter Patient Data for Treatment Response Prediction')
    input_data_treatment = {}
    for feature in selected_features_treatment:
        input_data_treatment[feature] = st.text_input(feature)

    if st.button('Predict Treatment Response'):
        try:
            input_df_treatment = pd.DataFrame([input_data_treatment], columns=selected_features_treatment)
            input_df_treatment = pd.get_dummies(input_df_treatment)
            missing_cols_treatment = set(X_train_treatment.columns) - set(input_df_treatment.columns)
            for col in missing_cols_treatment:
                input_df_treatment[col] = 0
            input_df_treatment = input_df_treatment[X_train_treatment.columns]
            prediction_treatment = model_treatment.predict(input_df_treatment)
            st.success(f'Predicted treatment response: {prediction_treatment[0]}')
            
            # Generate SHAP explanation
            explainer_treatment, shap_values_treatment = generate_shap_explanation(
                model_treatment, X_train_treatment, input_df_treatment, model_type='tree'
            )
            
            if explainer_treatment is not None and shap_values_treatment is not None:
                display_shap_plots(explainer_treatment, shap_values_treatment, input_df_treatment, X_train_treatment.columns)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

elif page == "Chemotherapy Prediction":
    st.title('Chemotherapy Prediction')

    selected_features_chemo = [
        "Age recode with <1 year olds and 90+",
        "Sex",
        "Race recode (W, B, AI, API)",
        "Primary Site - labeled",
        "Radiation recode",
        "RX Summ--Systemic/Sur Seq (2007+)",
        "RX Summ--Surg/Rad Seq",
        "Histologic Type ICD-O-3",
        "Months from diagnosis to treatment",
        "Grade Recode (thru 2017)",
        "Summary stage 2000 (1998-2017)",
        "Reason no cancer-directed surgery",
        "Regional nodes examined (1988+)",
        "SEER Combined Mets at DX-lung (2010+)"
    ]
    target_variable_chemo = 'Chemotherapy recode (yes, no/unk)'  # Replace with correct target variable

    X_chemo, y_chemo = preprocess_data(treatment_chemo, selected_features_chemo, target_variable_chemo)
    X_train_chemo, X_test_chemo, y_train_chemo, y_test_chemo = train_test_split(X_chemo, y_chemo, test_size=0.2, random_state=42)

    model_chemo, _ = train_logistic_regression(X_train_chemo, y_train_chemo)

    # User input form for chemotherapy prediction
    st.subheader('Enter Patient Data for Chemotherapy Prediction')
    input_data_chemo = {}
    for feature in selected_features_chemo:
        input_data_chemo[feature] = st.text_input(feature)

    if st.button('Predict Chemotherapy'):
        try:
            input_df_chemo = pd.DataFrame([input_data_chemo], columns=selected_features_chemo)
            input_df_chemo = pd.get_dummies(input_df_chemo)
            missing_cols_chemo = set(X_train_chemo.columns) - set(input_df_chemo.columns)
            for col in missing_cols_chemo:
                input_df_chemo[col] = 0
            input_df_chemo = input_df_chemo[X_train_chemo.columns]
            
            # For logistic regression, we need to scale the input
            input_scaled_chemo = model_chemo[1].transform(input_df_chemo) if model_chemo[1] else input_df_chemo
            prediction_chemo = model_chemo[0].predict(input_scaled_chemo)
            st.success(f'Predicted chemotherapy response: {prediction_chemo[0]}')
            
            # Generate SHAP explanation
            explainer_chemo, shap_values_chemo = generate_shap_explanation(
                model_chemo[0], X_train_chemo, input_scaled_chemo, model_type='linear', scaler=model_chemo[1]
            )
            
            if explainer_chemo is not None and shap_values_chemo is not None:
                display_shap_plots(explainer_chemo, shap_values_chemo, input_df_chemo, X_train_chemo.columns)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
