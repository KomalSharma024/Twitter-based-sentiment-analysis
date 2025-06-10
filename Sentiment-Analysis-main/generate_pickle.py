from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
import pickle

# Initialize the scaler and vectorizer
scaler = MinMaxScaler()
vectorizer = CountVectorizer()

# Save the scaler model
with open("Models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save the vectorizer model
with open("Models/Vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ scaler.pkl and Vectorizer.pkl generated successfully!")
