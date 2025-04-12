import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import streamlit as st

# Streamlit app title
st.title('Solar Energy Prediction')

# Function to load data
def load_data(file):
    return pd.read_csv(file)

# Function to preprocess data
def preprocess_data(data):
    # Assuming the date-time column is named 'DateTime'
   
    
    # Drop columns that cannot be converted to float
    for column in data.columns:
        if data[column].dtype == 'object':
            data = data.drop(column, axis=1)
    
    return data

# Upload the dataset
uploaded_file = st.file_uploader("Upload your solar prediction dataset (CSV file)", type=["csv"])
if uploaded_file is not None:
    data = load_data(uploaded_file)
    
    st.write("### Dataset")
    st.write(data.head())

    # Preprocess the data
    data = preprocess_data(data)

    # Define features and targets
    available_targets = [col for col in ['Radiation', 'Temperature', 'Pressure', 'Humidity', 'WindDirection', 'WindSpeed'] if col in data.columns]
    features = data.drop(available_targets, axis=1)
    target_data = data[available_targets]

    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Initialize dictionaries for storing models and predictions
    models = {}
    predictions = {}
    cv_scores_dict = {}

    # Loop through each target to select best features and train the model
    for target in available_targets:
        # Select top features for each target
        best_features = SelectKBest(score_func=f_regression, k='all').fit_transform(scaled_features, target_data[target])

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(best_features, target_data[target], test_size=0.2, random_state=42)

        # Train the model with cross-validation
        model = XGBRegressor()
        cv = KFold(n_splits=5, random_state=42, shuffle=True)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
        
        model.fit(X_train, y_train)
        models[target] = model
        predictions[target] = model.predict(X_test)
        cv_scores_dict[target] = cv_scores

        # Display cross-validation scores
        st.write(f'### Cross-Validation for {target}')
        st.write(f'Cross-Validation R^2 Scores: {cv_scores}')
        st.write(f'Mean R^2 Score: {np.mean(cv_scores)}')

    # Calculate and display performance metrics for each feature
    for target in available_targets:
        mse = mean_squared_error(y_test, predictions[target])
        r2 = r2_score(y_test, predictions[target])
        st.write(f'### Model Performance for {target}')
        st.write(f'Mean Squared Error: {mse}')
        st.write(f'R^2 Score: {r2}')

        # Display actual vs predicted values
        st.write(f'### Actual vs Predicted {target}')
        results_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions[target]})
        st.write(results_df.head())

        # Plot actual vs predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, predictions[target], alpha=0.5)
        plt.xlabel(f'Actual {target}')
        plt.ylabel(f'Predicted {target}')
        plt.title(f'Actual vs Predicted {target}')
        st.pyplot(plt)

    # Display feature importances for each model
    for target in available_targets:
        st.write(f'### Feature Importances for {target}')
        feature_importances = models[target].feature_importances_
        features_df = pd.DataFrame({'Feature': features.columns, 'Importance': feature_importances})
        features_df = features_df.sort_values(by='Importance', ascending=False)
        st.write(features_df)

    # Allow user to upload new data for predictions
    new_uploaded_file = st.file_uploader("Upload new data for predictions (CSV file)", type=["csv"], key="new_data")
    if new_uploaded_file is not None:
        new_data = load_data(new_uploaded_file)
        new_data = preprocess_data(new_data)
        new_scaled_features = scaler.transform(new_data)

        # Predict for each target
        new_predictions = {}
        for target in available_targets:
            # Select top features for the new data
            new_best_features = SelectKBest(score_func=f_regression, k='all').fit_transform(new_scaled_features, target_data[target])
            new_predictions[target] = models[target].predict(new_best_features)

        st.write('### New Predictions')
        for target in available_targets:
            st.write(f'Predictions for {target}')
            st.write(new_predictions[target])

            # Optionally plot new predictions
            st.write(f'### Plot New Predictions for {target}')
            plt.figure(figsize=(10, 6))
            plt.plot(new_predictions[target], label=f'Predicted {target}')
            plt.legend()
            plt.xlabel('Sample Index')
            plt.ylabel(f'Predicted {target}')
            plt.title(f'Predicted {target} for New Data')
            st.pyplot(plt)

else:
    st.write("Please upload a CSV file to proceed.")
