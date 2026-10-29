import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

print("Generating expanded synthetic dataset for 12 hand signs...")

classes = [
    'Hello', 'Thank You', 'I Love You', 'Peace', 
    'Thumbs Up', 'Thumbs Down', 'OK', 'Stop', 
    'Fist', 'Point', 'Three', 'Four'
]
samples_per_class = 350

data = []
labels = []

for sign in classes:
    for _ in range(samples_per_class):
        landmarks = np.zeros((21, 3))
        landmarks[0] = [0.5, 0.8, 0.0]  # Wrist base
        
        if sign == 'Hello':
            for finger in range(5):
                base = 1 + finger * 4
                for j in range(4):
                    landmarks[base + j] = [0.3 + finger * 0.1, 0.6 - j * 0.12, -0.01 * j]

        elif sign == 'I Love You':
            landmarks[1:5] = [[0.3, 0.65, 0], [0.25, 0.6, 0], [0.2, 0.55, 0], [0.15, 0.5, 0]]  # Thumb
            landmarks[5:9] = [[0.4, 0.6, 0], [0.4, 0.4, 0], [0.4, 0.25, 0], [0.4, 0.1, 0]]    # Index
            landmarks[9:13] = [[0.5, 0.6, 0], [0.5, 0.68, 0], [0.5, 0.72, 0], [0.5, 0.76, 0]] # Middle down
            landmarks[13:17] = [[0.6, 0.6, 0], [0.6, 0.68, 0], [0.6, 0.72, 0], [0.6, 0.76, 0]] # Ring down
            landmarks[17:21] = [[0.7, 0.6, 0], [0.7, 0.4, 0], [0.7, 0.25, 0], [0.7, 0.1, 0]]  # Pinky

        elif sign == 'Peace':
            landmarks[5:9] = [[0.4, 0.6, 0], [0.37, 0.4, 0], [0.34, 0.25, 0], [0.31, 0.1, 0]] # Index angle left
            landmarks[9:13] = [[0.5, 0.6, 0], [0.53, 0.4, 0], [0.56, 0.25, 0], [0.59, 0.1, 0]] # Middle angle right
            landmarks[13:21] = 0.7 # Others down

        elif sign == 'Thumbs Up':
            landmarks[1:5] = [[0.5, 0.6, 0], [0.5, 0.45, 0], [0.5, 0.3, 0], [0.5, 0.15, 0]]  # Thumb up
            for finger in range(1, 5):
                base = 1 + finger * 4
                landmarks[base:base+4] = [[0.5 + finger*0.05, 0.7, 0.1]] * 4  # Curled fingers

        elif sign == 'Thumbs Down':
            landmarks[1:5] = [[0.5, 0.8, 0], [0.5, 0.88, 0], [0.5, 0.94, 0], [0.5, 1.0, 0]]  # Thumb down
            for finger in range(1, 5):
                base = 1 + finger * 4
                landmarks[base:base+4] = [[0.5 + finger*0.05, 0.6, 0.1]] * 4  # Curled fingers

        elif sign == 'OK':
            landmarks[1:5] = [[0.42, 0.5, 0], [0.44, 0.45, 0], [0.45, 0.4, 0], [0.45, 0.35, 0]] # Thumb tip to index
            landmarks[5:9] = [[0.48, 0.5, 0], [0.46, 0.45, 0], [0.45, 0.4, 0], [0.45, 0.35, 0]] # Index tip touches thumb
            for finger in range(2, 5): # Middle, Ring, Pinky extended up
                base = 1 + finger * 4
                for j in range(4):
                    landmarks[base + j] = [0.5 + (finger-2)*0.08, 0.5 - j*0.12, 0]

        elif sign == 'Stop':
            for finger in range(5):
                base = 1 + finger * 4
                for j in range(4):
                    landmarks[base + j] = [0.32 + finger * 0.09, 0.55 - j * 0.14, -0.05]

        elif sign == 'Fist':
            for finger in range(5):
                base = 1 + finger * 4
                for j in range(4):
                    landmarks[base + j] = [0.45 + finger * 0.03, 0.7, 0.05 * j]

        elif sign == 'Point':
            landmarks[5:9] = [[0.5, 0.6, 0], [0.5, 0.4, 0], [0.5, 0.25, 0], [0.5, 0.1, 0]]  # Only Index extended
            for finger in [0, 2, 3, 4]:
                base = 1 + finger * 4
                landmarks[base:base+4] = [[0.5 + finger*0.03, 0.72, 0.05]] * 4

        elif sign == 'Three':
            landmarks[1:5] = [[0.3, 0.6, 0], [0.22, 0.5, 0], [0.15, 0.45, 0], [0.08, 0.4, 0]] # Thumb
            landmarks[5:9] = [[0.4, 0.6, 0], [0.4, 0.4, 0], [0.4, 0.25, 0], [0.4, 0.1, 0]]   # Index
            landmarks[9:13] = [[0.5, 0.6, 0], [0.5, 0.4, 0], [0.5, 0.25, 0], [0.5, 0.1, 0]]  # Middle
            landmarks[13:21] = 0.7  # Ring & Pinky down

        elif sign == 'Four':
            for finger in range(1, 5): # Index, Middle, Ring, Pinky extended
                base = 1 + finger * 4
                for j in range(4):
                    landmarks[base + j] = [0.38 + (finger-1)*0.08, 0.55 - j * 0.12, 0]
            landmarks[1:5] = [[0.5, 0.7, 0.1]] * 4 # Thumb tucked inside

        elif sign == 'Thank You':
            for finger in range(5):
                base = 1 + finger * 4
                for j in range(4):
                    landmarks[base + j] = [0.35 + finger * 0.08, 0.5 - j * 0.1, -0.1 - j * 0.05]

        # Add camera angle noise variation
        noise = np.random.normal(0, 0.018, landmarks.shape)
        landmarks += noise
        
        data.append(landmarks.flatten())
        labels.append(sign)

# Save pre-trained dataset to CSV
df = pd.DataFrame(data)
df['label'] = labels
df.to_csv('sign_language_dataset.csv', index=False)

# Train Random Forest AI model
X = df.drop('label', axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=120, random_state=42)
model.fit(X_train, y_train)

# Save trained model
with open('sign_language_model.pkl', 'wb') as f:
    pickle.dump(model, f)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"Dataset generated for 12 signs! Model accuracy: {acc * 100:.2f}%")