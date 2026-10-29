import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd
import urllib.request
import os

# 1. Download model if missing
task_path = 'hand_landmarker.task'
if not os.path.exists(task_path):
    print("Downloading AI model file (only happens once)...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, task_path)
    print("Download complete!")

# 2. Setup the new MediaPipe Tasks API
base_options = python.BaseOptions(model_asset_path=task_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)

# Define the signs you want to teach the AI
classes = ['Hello', 'Thank You', 'I Love You']
num_samples_per_class = 200 

data = []
labels = []

cap = cv2.VideoCapture(0)
frame_timestamp = 0 # MediaPipe needs a timeline to track video

for sign_name in classes:
    print(f"Ready to record '{sign_name}'. Press 'r' to start.")
    while True:
        ret, frame = cap.read()
        if not ret: continue
        cv2.putText(frame, f"Press 'r' to record: {sign_name}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Data Collection", frame)
        if cv2.waitKey(1) & 0xFF == ord('r'):
            break

    print(f"Recording {sign_name}...")
    samples_collected = 0
    
    while samples_collected < num_samples_per_class:
        ret, frame = cap.read()
        if not ret: continue
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        frame_timestamp += 33 # Simulating ~30 FPS timeline
        result = detector.detect_for_video(mp_image, frame_timestamp)
        
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                row = []
                h, w, c = frame.shape
                for landmark in hand_landmarks:
                    row.extend([landmark.x, landmark.y, landmark.z])
                    # Draw a green dot on each joint
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                
                data.append(row)
                labels.append(sign_name)
                samples_collected += 1
                
        cv2.putText(frame, f"Recording {sign_name}: {samples_collected}/{num_samples_per_class}", 
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Data Collection", frame)
        cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(data)
df['label'] = labels
df.to_csv('sign_language_dataset.csv', index=False)
print("Data collection complete. Saved to sign_language_dataset.csv")