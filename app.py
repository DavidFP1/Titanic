import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load('titanic_final_model.pkl')

# Define function for predictions
def predict_survival(sex, fare, age, pclass, family_size):
    data = pd.DataFrame({'Sex': [sex], 'Fare': [fare], 'Age': [age], 'Pclass': [pclass], 'FamilySize': [family_size]})
    prediction = model.predict(data)
    return "Survived" if prediction[0] == 1 else "Did Not Survive"

# Streamlit UI
st.title("🚢 Titanic Survival Prediction Dashboard")

# User Inputs
sex = st.radio("Sex", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
fare = st.slider("Fare Paid", min_value=0.0, max_value=512.0, value=30.0)
age = st.slider("Age", min_value=0, max_value=80, value=30)
pclass = st.selectbox("Passenger Class", [1, 2, 3])
family_size = st.slider("Family Size", min_value=1, max_value=10, value=1)

# Prediction Button
if st.button("Predict"):
    result = predict_survival(sex, fare, age, pclass, family_size)
    st.success(f"Prediction: **{result}**")

