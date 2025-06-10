import streamlit as st
import pandas as pd
import requests

# API Endpoint (Ensure your Flask app is running at this address)
prediction_endpoint = "http://127.0.0.1:5000/predict"

st.title("Text Sentiment Predictor")

# --- Single Text Input ---
user_input = st.text_input("Enter text and click on Predict", "")

if st.button("Predict"):
    if user_input:
        data = {"text": user_input}
        response = requests.post("http://127.0.0.1:5000/predict", json=data)

        if response.status_code == 200:
            result = response.json()
            prediction = result.get("prediction")
            probability = result.get("probability")
            st.success(f"Prediction: {prediction.capitalize()} (Confidence: {probability:.2f})")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    else:
        st.warning("Please enter some text for prediction.")


# --- Bulk Prediction from CSV ---
st.markdown("---")
uploaded_file = st.file_uploader("Upload a CSV file for bulk prediction", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        if "text" in df.columns:
            df.dropna(subset=["text"], inplace=True)  # Drop missing text rows
            st.write("Uploaded Data Preview:", df.head())

            with st.spinner("Sending data for prediction..."):
                response = requests.post(prediction_endpoint, json={"texts": df["text"].tolist()})

            if response.status_code == 200:
                predictions = response.json().get("predictions", [])
                if predictions:
                    df["Prediction"] = predictions
                    st.success("Predictions added successfully!")
                    st.write(df)
                    # Optionally, provide download link
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
                else:
                    st.warning("Received empty predictions.")
            else:
                st.error(f"Error {response.status_code}: {response.text}")
        else:
            st.error("CSV must contain a 'text' column.")

    except Exception as e:
        st.error(f"Could not process the uploaded file: {e}")
