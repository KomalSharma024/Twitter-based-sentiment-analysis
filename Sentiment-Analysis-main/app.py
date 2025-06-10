import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import os

# ---------- Function to Save to History ----------
def save_to_history(tweet, sentiment):
    file_name = "history.csv"
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
    else:
        df = pd.DataFrame(columns=["Tweet", "Sentiment"])

    new_entry = {"Tweet": tweet, "Sentiment": sentiment}
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df.to_csv(file_name, index=False)

# ---------- Landing Page ----------
def show_landing():
    st.title("📅 Welcome to Sentiment Analyzer")
    st.write("This web app analyzes the sentiment of tweets (Positive, Neutral, Negative).")
    st.image("landing.jpg", use_container_width=True)

# ---------- History Page ----------
def show_history():
    st.subheader("📊 Sentiment Analysis History")

    try:
        df = pd.read_csv("history.csv")

        if df.empty:
            st.warning("History is empty.")
            return

        st.dataframe(df)

        # Plot bar chart
        fig, ax = plt.subplots()
        sns.set_style("whitegrid")
        sns.countplot(x='Sentiment', data=df, palette='pastel', ax=ax)
        ax.set_title("Sentiment Count")
        st.pyplot(fig)

        # Pie chart
        fig1, ax1 = plt.subplots()
        df['Sentiment'].value_counts().plot.pie(
            autopct='%1.1f%%', ax=ax1, colors=['skyblue', 'lightcoral', 'lightgreen']
        )
        ax1.set_ylabel('')
        ax1.set_title('Sentiment Distribution')
        st.pyplot(fig1)

    except FileNotFoundError:
        st.error("No history file found.")

# ---------- Analyze Page ----------
def analyze_sentiment():
    st.subheader("🧠 Analyze a Tweet")

    user_input = st.text_area("Enter a tweet here:")

    if st.button("Analyze"):
        if user_input:
            try:
                api_url = "http://127.0.0.1:5000/predict"  # Backend URL
                response = requests.post(api_url, json={"text": user_input})

                emoji_map = {
                    "positive": "😊",
                    "negative": "😞",
                    "neutral": "😐"
                }

                if response.status_code == 200:
                    response_json = response.json()
                    prediction = response_json.get("prediction", "").lower()

                    if prediction in emoji_map:
                        predicted_label = prediction.capitalize()
                        emoji = emoji_map[prediction]
                        st.success(f"Predicted Sentiment: **{predicted_label} {emoji}**")

                        # Save to history
                        save_to_history(user_input, predicted_label)
                    else:
                        st.error("Unexpected prediction received from API.")

                else:
                    st.error("Prediction failed. Server returned an error.")

            except requests.exceptions.ConnectionError:
                st.error("⚠️ Backend server not running at http://127.0.0.1:5000. Start it first.")
        else:
            st.warning("Please enter a tweet before clicking Analyze.")

# ---------- Modern Sign In / Sign Up ----------
def show_sign_in_signup():
    st.markdown("""
        <style>
            .auth-container {
                max-width: 400px;
                margin: auto;
                padding: 2rem;
                border-radius: 15px;
                background-color: #f9f9f9;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            .auth-title {
                text-align: center;
                font-size: 2rem;
                font-weight: bold;
                margin-bottom: 10px;
            }
            .auth-subtitle {
                text-align: center;
                color: gray;
                font-size: 0.95rem;
                margin-bottom: 25px;
            }
        </style>
        <div class='auth-container'>
    """, unsafe_allow_html=True)

    selected_tab = st.radio("Choose an option:", ("🔓 Sign In", "🆕 Sign Up"), horizontal=True)

    st.markdown(f"<div class='auth-title'>{'Welcome Back!' if 'Sign In' in selected_tab else 'Create Your Account'}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='auth-subtitle'>{'Sign in to continue' if 'Sign In' in selected_tab else 'Start your journey with us'}</div>", unsafe_allow_html=True)

    if 'Sign In' in selected_tab:
        username = st.text_input("👤 Username", placeholder="Enter your username")
        password = st.text_input("🔑 Password", type="password", placeholder="Enter your password")

        if st.button("Sign In"):
            if username and password:
                st.session_state.signed_in = True
                st.success("✅ Successfully signed in!")
                st.rerun()
            else:
                st.error("❗ Please enter both username and password.")
    else:
        username = st.text_input("👤 Choose a Username", placeholder="Create a username")
        email = st.text_input("📧 Email", placeholder="Enter your email")
        password = st.text_input("🔑 Choose a Password", type="password", placeholder="Create a password")

        if st.button("Sign Up"):
            if username and email and password:
                st.session_state.signed_in = True
                st.success("🎉 Account created successfully!")
                st.rerun()
            else:
                st.error("❗ Please fill all fields to sign up.")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Main Navigation ----------
def main_app():
    if 'signed_in' not in st.session_state:
        st.session_state.signed_in = False

    if not st.session_state.signed_in:
        show_sign_in_signup()
    else:
        st.sidebar.title("Navigation")
        selected_option = st.sidebar.radio("Go to", ["Home", "Analyze", "History", "About Us", "Contact Us", "Logout"])

        if selected_option == "Home":
            show_landing()
        elif selected_option == "Analyze":
            analyze_sentiment()
        elif selected_option == "History":
            show_history()
        elif selected_option == "About Us":
            st.subheader("📌 About Us")
            st.write("""
                Welcome to our Sentiment Analysis Web Application! 🌟

                Our mission is to empower users with real-time sentiment insights by leveraging the power of Machine Learning and Natural Language Processing (NLP).

                ### 🚀 Key Features:
                - Analyze tweets for Positive, Negative, and Neutral sentiments.
                - Visualize the mood with charts and graphs.
                - Track past analyses easily.
                - Friendly and responsive support.

                ### 👩‍💻 About the Developer:
                Developed by **Komal Sharma** and Team!  
                Passionate about AI, ML, and simplifying technology for everyone.

                ### 📬 Connect With Us:
                - 📞 **Personal Contact**: +91-6387642442
                - 📞 **Group Contact**: +91-8738832086
                - 📧 **Email**: komal993596@gmail.com
                - 🌐 LinkedIn: [Click Here](https://www.linkedin.com/)
                - 🌐 GitHub: [Click Here](https://github.com/)

                Thank you for visiting! 😊
            """)
        elif selected_option == "Contact Us":
            st.subheader("📞 Contact Us")
            st.write("""
                For any queries, feel free to reach out to us:  

                📞 **Personal Contact**: +91-6387642442  
                📞 **Group Contact**: +91-8738832086  
                📧 **Email**: komal993596@gmail.com  

                We are happy to assist you! 🙌
            """)
        elif selected_option == "Logout":
            st.session_state.signed_in = False
            st.success("Logged out successfully!")
            st.rerun()

# ---------- Run the App ----------
if __name__ == "__main__":
    main_app()

