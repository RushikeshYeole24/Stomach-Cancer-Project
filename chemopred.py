import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Load your dataset
your_dataset = pd.read_csv('treatmentresponse.csv')

# Convert all column names to strings
your_dataset.columns = your_dataset.columns.astype(str)

selected_features = [
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

target_variable = "RX Summ--Surg/Rad Seq"  # Replace this with the actual target variable from your dataset

X = your_dataset[selected_features]
y = your_dataset[target_variable]

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='most_frequent')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Separate numeric and categorical features
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

# One-hot encode categorical features
encoder = OneHotEncoder(handle_unknown='ignore')
X_train_encoded = encoder.fit_transform(X_train[categorical_features])
X_test_encoded = encoder.transform(X_test[categorical_features])

# Concatenate encoded features with numeric features
X_train_final = pd.concat([pd.DataFrame(X_train_encoded.toarray()), X_train[numeric_features].reset_index(drop=True)], axis=1)
X_test_final = pd.concat([pd.DataFrame(X_test_encoded.toarray()), X_test[numeric_features].reset_index(drop=True)], axis=1)

# Initialize and train a Gradient Boosting Classifier
clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_final, y_train)

# Predict on the test set
y_pred = clf.predict(X_test_final)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

# Streamlit code
import streamlit as st

# Sidebar navigation
page = st.sidebar.selectbox("Navigate to", ["Home", "Stage Prediction", "Survival Prediction", "Treatment Response"])

if page == "Home":
    st.subheader('Welcome to Cancer Prediction System')
    st.write('This system helps predict cancer stages, survival status, and treatment response based on patient data.')

elif page == "Stage Prediction":
    # Your existing code for stage prediction
    pass

elif page == "Survival Prediction":
    # Your existing code for survival prediction
    pass

elif page == "Treatment Response":
    st.title('Treatment Response Prediction')

    # User input form for treatment response prediction
    input_data_treatment = {}
    for feature in selected_features:
        input_data_treatment[feature] = st.text_input(feature)

    if st.button('Predict Treatment Response'):
        input_df_treatment = pd.DataFrame([input_data_treatment], columns=selected_features)
        input_df_treatment = pd.get_dummies(input_df_treatment)
        missing_cols_treatment = set(X_train_final.columns) - set(input_df_treatment.columns)
        for col in missing_cols_treatment:
            input_df_treatment[col] = 0
        input_df_treatment = input_df_treatment[X_train_final.columns]
        prediction_treatment = clf.predict(input_df_treatment)
        st.success(f'Predicted treatment response: {prediction_treatment[0]}')

