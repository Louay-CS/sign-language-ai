import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import urllib.request
import os
import pickle
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display

import tkinter as tk
from tkinter import simpledialog

# Setup popup window
root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
task_path = 'hand_landmarker.task'
if not os.path.exists(task_path):
    print("Downloading hand tracking model...")
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", task_path)

base_options = python.BaseOptions(model_asset_path=task_path)
# --- UPGRADED: Now tracks BOTH hands for maximum accuracy ---
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2, running_mode=vision.RunningMode.VIDEO)
detector = vision.HandLandmarker.create_from_options(options)

DATASET_FILE = 'my_custom_signs.pkl'
X_data, y_labels, ai_model = [], [], None

if os.path.exists(DATASET_FILE):
    with open(DATASET_FILE, 'rb') as f:
        data = pickle.load(f)
        X_data, y_labels = data['X'], data['y']
    if len(X_data) > 0:
        ai_model = KNeighborsClassifier(n_neighbors=3)
        ai_model.fit(X_data, y_labels)

DICTIONARY = {
    "HELLO": "أهلاً", "PEACE": "سلام", "WATER": "ماء", "FOUR": "أربعة", 
    "ONE": "واحد", "CALL ME": "اتصل بي", "I LOVE YOU": "أحبك", "OK": "حسناً", 
    "THUMBS UP": "ممتاز", "THUMBS DOWN": "سيء", "FIST": "قبضة", "HELP": "مساعدة",
    "THANK YOU": "شكراً", "YES": "نعم", "NO": "لا", "PLEASE": "من فضلك"
}

# ==========================================
# 2. GEOMETRY & ML HELPERS (DUAL HAND)
# ==========================================
def dist(p1, p2): return np.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2 + (p1.z-p2.z)**2)

def extract_features(hand_landmarks):
    wrist = hand_landmarks[0]
    scale = dist(hand_landmarks[9], wrist)
    if scale == 0: scale = 1.0
    features = []
    for lm in hand_landmarks:
        features.extend([(lm.x - wrist.x)/scale, (lm.y - wrist.y)/scale, (lm.z - wrist.z)/scale])
    return np.array(features)

def evaluate_basics(hands_landmarks):
    # 1. Check for two-handed signs first
    if len(hands_landmarks) == 2:
        wrist1, wrist2 = hands_landmarks[0][0], hands_landmarks[1][0]
        if dist(wrist1, wrist2) < 0.25:
            return "HELP"

    # 2. Check each hand independently (Left or Right, it doesn't matter!)
    for hand_lms in hands_landmarks:
        wrist = hand_lms[0]
        scale = dist(hand_lms[9], wrist)
        if scale == 0: scale = 1.0

        thumb_ext = dist(hand_lms[4], hand_lms[17]) > 0.85 * scale
        index_ext = dist(hand_lms[8], wrist) > dist(hand_lms[6], wrist)
        mid_ext   = dist(hand_lms[12], wrist) > dist(hand_lms[10], wrist)
        ring_ext  = dist(hand_lms[16], wrist) > dist(hand_lms[14], wrist)
        pinky_ext = dist(hand_lms[20], wrist) > dist(hand_lms[18], wrist)
        thumb_idx_d = dist(hand_lms[4], hand_lms[8]) / scale

        if index_ext and mid_ext and ring_ext and pinky_ext and thumb_ext: return "HELLO"
        if index_ext and mid_ext and ring_ext and pinky_ext and not thumb_ext: return "FOUR"
        if index_ext and mid_ext and ring_ext and not pinky_ext: return "WATER"
        if index_ext and mid_ext and not ring_ext and not pinky_ext: return "PEACE"
        if index_ext and not mid_ext and not ring_ext and not pinky_ext: return "ONE"
        if thumb_ext and pinky_ext and not index_ext and not mid_ext and not ring_ext: return "CALL ME"
        if thumb_ext and index_ext and pinky_ext and not mid_ext and not ring_ext: return "I LOVE YOU"
        if thumb_idx_d < 0.35 and mid_ext and ring_ext and pinky_ext: return "OK"
        if not index_ext and not mid_ext and not ring_ext and not pinky_ext:
            if (hand_lms[4].y - wrist.y) / scale < -0.8: return "THUMBS UP"
            if (hand_lms[4].y - wrist.y) / scale > 0.8: return "THUMBS DOWN"
            return "FIST"
            
    return None

# ==========================================
# 3. UI RENDERING ENGINE
# ==========================================
def render_pro_ui(frame, current_word, sentence, hold_progress):
    h, w, _ = frame.shape
    overlay = frame.copy()
    
    cv2.rectangle(overlay, (0, 0), (w, 100), (20, 20, 20), -1)
    cv2.rectangle(overlay, (0, h - 120), (w, h), (15, 15, 15), -1)
    frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

    if hold_progress > 0:
        bar_width = int((hold_progress / 20.0) * w)
        cv2.rectangle(frame, (0, h - 125), (bar_width, h - 120), (0, 255, 120), -1)

    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    try:
        font_ar_large = ImageFont.truetype("arial.ttf", 36)
        font_en = ImageFont.truetype("arial.ttf", 20)
    except:
        font_ar_large = font_en = ImageFont.load_default()

    en_sentence = " ".join(sentence)
    ar_sentence = " ".join([DICTIONARY.get(w, w) for w in sentence])
    
    draw.text((w//2, 35), get_display(arabic_reshaper.reshape(ar_sentence if ar_sentence else "الجملة...")), 
              font=font_ar_large, fill=(255, 215, 0), anchor="mm")
    draw.text((w//2, 75), f"{en_sentence if en_sentence else 'Sentence buffer...'}", 
              font=font_en, fill=(200, 200, 200), anchor="mm")

    ar_word = DICTIONARY.get(current_word, current_word)
    color = (0, 255, 120) if hold_progress >= 20 else (255, 255, 255)
    
    draw.text((w//2, h - 70), get_display(arabic_reshaper.reshape(ar_word)), font=font_ar_large, fill=color, anchor="mm")
    draw.text((w//2, h - 25), f"[{current_word}]", font=font_en, fill=(150, 150, 150), anchor="mm")

    draw.text((10, h - 110), "N: New Word\nC: Clear\nQ: Quit", font=font_en, fill=(100, 100, 100))

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ==========================================
# 4. MAIN LOOP
# ==========================================
cap = cv2.VideoCapture(0)
frame_timestamp = 0

sentence_buffer = []
candidate_word = None
hold_frames = 0
cooldown = 0
recording_mode = False
frames_recorded = 0
word_to_learn = ""

print("Dual-Hand Tracking Active! Use your Left OR Right hand.")

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    frame_timestamp += 33
    result = detector.detect_for_video(mp_image, frame_timestamp)
    
    detected = None

    if result.hand_landmarks:
        # Draw dots on ALL detected hands
        for hand_lms in result.hand_landmarks:
            for lm in hand_lms: 
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)

        # Recording Mode (Uses the primary hand it sees)
        if recording_mode:
            features = extract_features(result.hand_landmarks[0])
            X_data.append(features)
            y_labels.append(word_to_learn.upper())
            frames_recorded += 1
            cv2.putText(frame, f"Recording '{word_to_learn.upper()}': {frames_recorded}/40", (w//2 - 200, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
            if frames_recorded >= 40:
                recording_mode = False
                ai_model = KNeighborsClassifier(n_neighbors=3)
                ai_model.fit(X_data, y_labels)
                with open(DATASET_FILE, 'wb') as f: pickle.dump({'X': X_data, 'y': y_labels}, f)
        
        # Translation Mode
        else:
            detected = evaluate_basics(result.hand_landmarks)
            
            # If no basic sign was found, check all hands against the custom AI model
            if not detected and ai_model is not None:
                for hand_lms in result.hand_landmarks:
                    features = extract_features(hand_lms)
                    if np.max(ai_model.predict_proba([features])[0]) > 0.7:
                        detected = ai_model.predict([features])[0]
                        break # Stop looking if we found a match

    # Sentence Builder Logic
    if not recording_mode:
        if cooldown > 0:
            cooldown -= 1
        elif detected:
            if detected == candidate_word:
                hold_frames += 1
                if hold_frames == 20:
                    sentence_buffer.append(detected)
                    if len(sentence_buffer) > 6: sentence_buffer.pop(0)
                    cooldown = 45
            else:
                candidate_word = detected
                hold_frames = 1
        else:
            candidate_word = None
            hold_frames = 0

    # Key Controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('n') and not recording_mode:
        user_input = simpledialog.askstring("Teach New Sign", "Type the word you want to teach the AI:", parent=root)
        if user_input:
            word_to_learn = user_input.strip()
            recording_mode = True
            frames_recorded = 0
            
    elif key == ord('c'):
        sentence_buffer = []
    elif key == ord('q'):
        break

    if not recording_mode:
        frame = render_pro_ui(frame, candidate_word if candidate_word else "...", sentence_buffer, hold_frames if cooldown == 0 else 0)

    cv2.imshow("Sign Language Pro Translator", frame)

cap.release()
cv2.destroyAllWindows()