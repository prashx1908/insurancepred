import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

def main():
    st.title("Insurance Cost Prediction with Artificial Neural Network")
    st.sidebar.title("Options")

    # Read data
    @st.cache
    def load_data():
        return pd.read_csv("insurance.csv")

    data = load_data()

    # Encode categorical variables
    le = LabelEncoder()
    data['sex'] = le.fit_transform(data['sex'])
    data['smoker'] = le.fit_transform(data['smoker'])

    # One-hot encode 'region' variable
    onehot_encoder = OneHotEncoder()
    region_encoded = onehot_encoder.fit_transform(data[['region']])
    region_column_names = onehot_encoder.get_feature_names_out(['region'])

    # Ensure that the number of columns in region_encoded matches the number of columns in region_column_names
    if region_encoded.shape[1] != len(region_column_names):
        raise ValueError("Number of columns in one-hot encoded region data does not match the number of columns in region_column_names")

    data[region_column_names] = region_encoded.toarray()  # Convert to dense array
    data = data.drop(['region'], axis=1)

    # Split data into features and target
    X = data.drop("charges", axis=1)
    y = data["charges"]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build the model
    model = Sequential()
    model.add(Dense(units=512, input_dim=X_train_scaled.shape[1]))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(Dense(units=256))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(Dense(units=128))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(units=64))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(units=1, activation='linear'))

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mean_squared_error')

    # Define early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    # Train the model
    history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stopping], verbose=1)

    # Evaluate the model
    mse = model.evaluate(X_test_scaled, y_test)
    st.write(f'Mean Squared Error on Test Set: {mse}')

    # Make predictions
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    st.write(f'R-squared on Test Set: {r2}')

    # Input columns for prediction
    age = st.sidebar.slider("Age", min_value=18, max_value=100, value=25)
    sex = st.sidebar.radio("Sex", ["Male", "Female"])
    bmi = st.sidebar.slider("BMI (Body Mass Index)", min_value=10.0, max_value=50.0, value=22.0, step=0.1)
    children = st.sidebar.slider("Number of Children", min_value=0, max_value=10, value=2)
    smoker = st.sidebar.radio("Smoker", ["No", "Yes"])
    region = st.sidebar.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

    # Convert inputs to model format
    sex = 1 if sex == "Male" else 0
    smoker = 1 if smoker == "Yes" else 0
    region_index = ["Northeast", "Northwest", "Southeast", "Southwest"].index(region)
    region_encoded = [0] * 4
    region_encoded[region_index] = 1

    input_data = [[age, sex, bmi, children, smoker] + region_encoded]
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)[0][0]

    st.sidebar.subheader("Predicted Insurance Charges:")
    st.sidebar.write(prediction)

    # Plot feature importances
    weights = model.layers[0].get_weights()[0]
    feature_importances = np.sum(np.abs(weights), axis=1)
    feature_names = X.columns
    indices = np.argsort(feature_importances)[::-1]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(range(X.shape[1]), feature_importances[indices], align="center")
    ax.set_xticks(range(X.shape[1]))
    ax.set_xticklabels(feature_names[indices], rotation=45)
    ax.set_title("Feature Importances - ANN")
    st.pyplot(fig)


if __name__ == "__main__":
    main()
