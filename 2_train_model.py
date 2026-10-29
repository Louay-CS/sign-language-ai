import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

print("Loading dataset...")
df = pd.read_csv('sign_language_dataset.csv')

# Split features (X) and labels (y)
X = df.drop('label', axis=1)
y = df['label']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training the AI model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test the accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model trained with accuracy: {accuracy * 100:.2f}%")

# Save the trained model
with open('sign_language_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved successfully as 'sign_language_model.pkl'.")