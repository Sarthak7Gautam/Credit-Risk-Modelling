import streamlit as st
import pandas as pd
import joblib

model = joblib.load(
    r"C:\Users\dell\OneDrive\Desktop\Credit Risk Modelling\Scripts\gradient_model.pkl"
)
feature_encoder = joblib.load(
    r"C:\Users\dell\OneDrive\Desktop\Credit Risk Modelling\Scripts\full_preprocessing.pkl"
)

st.title("Credit Risk Modelling App")
st.write("Enter the details and press the predict button to see the Credit Risk")
st.divider()

age = st.number_input("Age", min_value=18, max_value=100, value=30)
job = st.selectbox(
    "Job (0: unskilled, 1: skilled, 2: highly skilled)", options=[0, 1, 2]
)
credit_amount = st.number_input("Credit amount", min_value=0, value=1000)
duration = st.number_input("Duration (in months)", min_value=1, value=12)

sex = st.selectbox("Sex", options=["male", "female"])
housing = st.selectbox("Housing Type", options=["free", "own", "rent"])
saving = st.selectbox(
    "Saving Account", options=["little", "moderate", "quite rich", "rich"]
)
checking = st.selectbox("Checking Account", options=["little", "moderate", "rich"])

raw_input_df = pd.DataFrame(
    {
        "Age": [age],
        "Sex": [sex],
        "Job": [job],
        "Housing": [housing],
        "Saving accounts": [saving],
        "Checking account": [checking],
        "Credit amount": [credit_amount],
        "Duration": [duration],
    }
)

if st.button("Predict Risk"):
    try:
        transformed_data = feature_encoder.transform(raw_input_df)
        prediction = model.predict(transformed_data)

        if prediction[0] == 1:
            st.error("The Predicted Risk is HIGH")
        else:
            st.success("The Predicted Risk is LOW")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
