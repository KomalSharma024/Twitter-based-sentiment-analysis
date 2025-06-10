import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import pickle

# Extended synthetic dataset (30 samples total)
data = {
    'text': [
        # Positive
        "I love this product!",
        "This is amazing, totally worth it.",
        "Excellent service, very satisfied.",
        "I highly recommend it.",
        "Best experience ever.",
        "Absolutely fantastic!",
        "Really happy with the purchase.",
        "It exceeded my expectations.",
        "Very impressed with the quality.",
        "I’m so pleased with this item.",

        # Negative
        "Terrible experience, I hate it.",
        "Worst purchase I’ve ever made.",
        "Not worth the money.",
        "Very disappointing.",
        "I want a refund.",
        "Product was broken.",
        "Service was horrible.",
        "Extremely dissatisfied.",
        "It didn't work at all.",
        "Totally useless.",

        # Neutral
        "It was okay, nothing special.",
        "I don't have much to say.",
        "It worked as expected.",
        "Neither good nor bad.",
        "It’s fine, just average.",
        "Decent product, could be better.",
        "I received the item.",
        "Okay service.",
        "Neutral experience.",
        "Delivery was on time."
    ],
    'label': [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,      # Positive = 1
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,      # Negative = 0
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2       # Neutral = 2
    ]
}

df = pd.DataFrame(data)

# Split data - stratify to keep class distribution
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Create pipeline (TF-IDF + Logistic Regression)
model = make_pipeline(
    TfidfVectorizer(),
    LogisticRegression(max_iter=1000, random_state=42)
)

# Train model
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

print("Classification Report on Test Set:")
print(classification_report(y_test, y_pred, target_names=["negative", "positive", "neutral"]))

# Save model pipeline
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved successfully!")
