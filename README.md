\# Real-Time Bilingual Sign Language AI 🤟



A real-time, dual-hand Sign Language Translator built with Python, OpenCV, and MediaPipe. 

It features a \*\*Hybrid Translation Engine\*\* that instantly translates basic ASL geometry 

while using a custom K-Nearest Neighbors (KNN) machine learning model to let users teach 

the AI an unlimited number of custom signs in real-time.



\## Features

\* \*\*Dual-Hand Tracking:\*\* Tracks left and right hands simultaneously.

\* \*\*Universal Teachable Engine:\*\* Press 'N' to open a popup, type a word, and train the AI on a new sign in 1 second.

\* \*\*Sentence Maker UI:\*\* Holds words and dynamically builds sentences using a visual progress bar.

\* \*\*Bilingual Display:\*\* Renders Right-to-Left Arabic text and English simultaneously using PIL and `arabic-reshaper`.



\## Tech Stack

\* \*\*Computer Vision:\*\* OpenCV, MediaPipe Tasks API

\* \*\*Machine Learning:\*\* Scikit-Learn (KNeighborsClassifier)

\* \*\*UI/Rendering:\*\* Pillow, Tkinter, python-bidi



\## How to Run

1\. Install dependencies: `pip install opencv-python mediapipe scikit-learn Pillow arabic-reshaper python-bidi`

2\. Run the engine: `python 3\_realtime\_translator.py`

