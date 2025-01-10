import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset
# Assuming you have a CSV file named 'stagepred.csv'
data = pd.read_csv('stagepred.csv')

# Select relevant features and target variable
selected_features = [
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
target_variable = 'Combined Summary Stage (2004+)'

# Drop rows with missing values
data = data.dropna(subset=selected_features + [target_variable])

# Separate features and target variable
X = data[selected_features]
y = data[target_variable]

# Convert categorical variables to dummy variables
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the decision tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Take user input and make predictions
def predict_status(input_data, selected_features, X_train, scaler, model):
    # Convert input data to DataFrame
    input_df = pd.DataFrame(input_data)

    # Convert categorical variables to dummy variables
    input_df = pd.get_dummies(input_df)

    # Add missing dummy columns using pd.concat(axis=1)
    missing_cols = set(X_train.columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0

    # Reorder columns to match X_train
    input_df = input_df[X_train.columns]

    # Standardize the features
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    return prediction[0]

# Streamlit app
st.title('Cancer Prediction System')

# Sidebar navigation
page = st.sidebar.selectbox("Navigate to", ["Home", "Predictions"])

if page == "Home":
    st.subheader('Welcome to Cancer Prediction System')
    st.write('This system helps predict cancer stages based on patient data.')

    # User input form
    with st.form(key='input_form'):
        input_data = {}
        for feature in selected_features:
            input_data[feature] = st.text_input(feature)
        submit_button = st.form_submit_button(label='Predict')

    # Display prediction result
    if submit_button:
        prediction = predict_status(input_data, selected_features, X_train, scaler, model)
        st.success(f'Predicted cancer stage: {prediction}')

elif page == "Predictions":
    # This section can remain as-is for your other predictions or models
    pass