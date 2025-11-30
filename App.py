import streamlit as st
import pandas as pd
import joblib
from src.model import train_models  # Optional: if you want to retrain from UI

# ✅ Load model and encoders
@st.cache_resource
def load_model():
    return joblib.load("models/best_model.pkl")

@st.cache_resource
def load_encoders():
    return joblib.load("models/label_encoders.pkl")

# ✅ Encode user input
def preprocess_input(user_input: dict, encoders: dict) -> pd.DataFrame:
    processed = user_input.copy()
    for col, encoder in encoders.items():
        processed[col] = encoder.transform([processed[col]])[0]
    return pd.DataFrame([processed])

# ✅ Predict survival
def predict_survival(input_df: pd.DataFrame, model) -> int:
    prediction = model.predict(input_df)[0]
    return prediction

# ✅ Streamlit UI
st.title("🚢 Titanic Survival Predictor")

st.markdown("Enter passenger details to predict survival:")

# Collect user input
sex = st.selectbox("Sex", ["male", "female"])
embarked = st.selectbox("Embarked", ["C", "Q", "S"])
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
age = st.slider("Age", 0, 100, 30)
fare = st.slider("Fare", 0.0, 500.0, 50.0)
sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)

user_input = {
    "Sex": sex,
    "Embarked": embarked,
    "Pclass": str(pclass),  # Ensure string for LabelEncoder
    "Age": age,
    "Fare": fare,
    "SibSp": sibsp,
    "Parch": parch
}

# Predict button
if st.button("Predict Survival"):
    model = load_model()
    encoders = load_encoders()
    input_df = preprocess_input(user_input, encoders)
    result = predict_survival(input_df, model)

    if result == 1:
        st.success("✅ Prediction: Survived")
    else:
        st.error("❌ Prediction: Did not survive")