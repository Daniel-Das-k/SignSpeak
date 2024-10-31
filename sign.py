import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
import googletrans
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pygame
# from tts import TextToSpeech

st.title('Sign Language Recognition')

st.sidebar.title("Navigation")
options = st.sidebar.selectbox("Choose the app mode", ["Sign Language Recognition", "Virtual Board"])

# Language selection
# language = st.selectbox("Select Language", ["en", "es", "fr", "hi"])

col1, col2 = st.columns([3, 2])
with col1:
    run = st.toggle('Run', value=True)
    FRAME_WINDOW = st.image([])
 
with col2:
    st.title("Output")
    output_text_area = st.subheader("")

# def tts(text, lang):
#     myobj = gTTS(text=text, lang=lang, slow=False)
#     myobj.save("audio.mp3")
#     pygame.mixer.init()
#     pygame.mixer.music.load("audio.mp3")
#     pygame.mixer.music.play()

actions = np.array(['hello', 'thanks', 'i love you'])

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils 

model.load_weights('actionMain.h5')

sequence = []
sentence = []
predictions = []
threshold = 0.5
temp = []
last_sentence = ""

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while run:
        ret, frame = cap.read()
        if not ret:
            continue
        
        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)

        if results.left_hand_landmarks or results.right_hand_landmarks:
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))
                
                if res[np.argmax(res)] > threshold:
                    detected_word = actions[np.argmax(res)]
                    if detected_word != last_sentence:
                        last_sentence = detected_word
                        sentence.append(detected_word)
                        output_text_area.write(detected_word)
                        # tts(detected_word, language)
                        print(detected_word)
                        temp.append(detected_word)

        FRAME_WINDOW.image(frame, channels="BGR")

        if not run:
            break

cap.release()
cv2.destroyAllWindows()