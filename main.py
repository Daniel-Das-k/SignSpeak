# import pickle
# import cv2
# import mediapipe as mp
# import numpy as np
# import streamlit as st
# from translate import TranslatorApp
# from tts import TextToSpeech
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense

# model_dict = pickle.load(open('./model.p', 'rb'))
# model = model_dict['model']
# translate = TranslatorApp()
# tts = TextToSpeech()

# actions = np.array(['hello', 'thanks', 'i love you'])
# lstm_model = Sequential()
# lstm_model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
# lstm_model.add(LSTM(128, return_sequences=True, activation='relu'))
# lstm_model.add(LSTM(64, return_sequences=False, activation='relu'))
# lstm_model.add(Dense(64, activation='relu'))
# lstm_model.add(Dense(32, activation='relu'))
# lstm_model.add(Dense(actions.shape[0], activation='softmax'))
# lstm_model.load_weights('actionMain.h5')

# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_holistic = mp.solutions.holistic

# labels_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '10'}

# st.set_page_config(layout="wide")
# st.title('Indian Sign Language and Number Recognition')

# option = st.sidebar.selectbox('Select Mode', ('Number Detection', 'Word Recognition'))

# col1, col2 = st.columns([3, 1])
# with col1:
#     run = st.checkbox('Run', value=True)
#     FRAME_WINDOW = st.image([])

# with col2:
#     st.title("Output")
#     output_text_area = st.empty()

# cap = cv2.VideoCapture(1)
# cap.set(3, 1280)
# cap.set(4, 720)

# def extract_features(hand_landmarks):
#     data_aux = []
#     x_ = []
#     y_ = []
#     for landmark in hand_landmarks.landmark:
#         x_.append(landmark.x)
#         y_.append(landmark.y)
#     for landmark in hand_landmarks.landmark:
#         data_aux.append(landmark.x - min(x_))
#         data_aux.append(landmark.y - min(y_))
#     if len(data_aux) > 42:
#         data_aux = data_aux[:42]
#     else:
#         data_aux.extend([0] * (42 - len(data_aux)))
#     return data_aux, x_, y_

# def mediapipe_detection(image, model):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image.flags.writeable = False
#     results = model.process(image)
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     return image, results

# def draw_styled_landmarks(image, results):
#     mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
#                               mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
#                               mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
#     mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
#                               mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
#                               mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
#     mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
#                               mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
#                               mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
#     mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
#                               mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
#                               mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

# def extract_keypoints(results):
#     pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
#     face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
#     lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
#     rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
#     return np.concatenate([pose, face, lh, rh])

# sequence = []
# sentence = []
# predictions = []
# last_detected_number = None
# last_detected_word = None
# prediction_buffer = []
# buffer_size = 5  

# while run:
#     ret, frame = cap.read()
#     if not ret:
#         continue

#     if option == 'Number Detection':
#         with mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.6) as hands:
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = hands.process(frame_rgb)
#             if results.multi_hand_landmarks:
#                 for hand_landmarks in results.multi_hand_landmarks:
#                     mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
#                                               mp_drawing_styles.get_default_hand_landmarks_style(), 
#                                               mp_drawing_styles.get_default_hand_connections_style())
#                     data_aux, x_, y_ = extract_features(hand_landmarks)
#                     prediction = model.predict([np.asarray(data_aux)])
#                     predicted_number = labels_dict[int(prediction[0])]

#                     prediction_buffer.append(predicted_number)
#                     if len(prediction_buffer) > buffer_size:
#                         prediction_buffer.pop(0)

#                     if prediction_buffer.count(predicted_number) > buffer_size // 2 and predicted_number != last_detected_number:
#                         last_detected_number = predicted_number
#                         content = translate.run(predicted_number)
#                         tts.run(content)
#                         output_text_area.subheader(predicted_number)
#                         print(predicted_number)
#                         x1 = int(min(x_) * frame.shape[1]) - 10
#                         y1 = int(min(y_) * frame.shape[0]) - 10
#                         x2 = int(max(x_) * frame.shape[1]) - 10
#                         y2 = int(max(y_) * frame.shape[0]) - 10
#                         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
#                         cv2.putText(frame, predicted_number, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)

#     elif option == 'Word Recognition':
#         with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#             image, results = mediapipe_detection(frame, holistic)
#             draw_styled_landmarks(image, results)
#             if results.left_hand_landmarks or results.right_hand_landmarks:
#                 keypoints = extract_keypoints(results)
#                 sequence.append(keypoints)
#                 sequence = sequence[-30:]
#                 if len(sequence) == 30:
#                     res = lstm_model.predict(np.expand_dims(sequence, axis=0))[0]
#                     detected_word = actions[np.argmax(res)]
#                     if detected_word != last_detected_word:
#                         last_detected_word = detected_word
#                         sentence.append(detected_word)
#                         content = translate.run(detected_word)
#                         tts.run(content)
#                         # pygame.mixer.init()
#                         # pygame.mixer.music.load("audio.mp3")
#                         # pygame.mixer.music.play()
#                         output_text_area.subheader(detected_word)
#                         print(detected_word)

#     FRAME_WINDOW.image(frame, channels="BGR")
#     if not run:
#         break

# cap.release()
# cv2.destroyAllWindows()

import pickle
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from translate import TranslatorApp
from tts import TextToSpeech
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load models and translators
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
translate = TranslatorApp()
tts = TextToSpeech()

# LSTM model for word recognition
actions = np.array(['hello', 'thanks', 'i love you'])
lstm_model = Sequential()
lstm_model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
lstm_model.add(LSTM(128, return_sequences=True, activation='relu'))
lstm_model.add(LSTM(64, return_sequences=False, activation='relu'))
lstm_model.add(Dense(64, activation='relu'))
lstm_model.add(Dense(32, activation='relu'))
lstm_model.add(Dense(actions.shape[0], activation='softmax'))
lstm_model.load_weights('actionMain.h5')

# Load one-hand and two-hand models for alphabet detection
one_hand_model_dict = pickle.load(open('./one_hand_model.p', 'rb'))
one_hand_model = one_hand_model_dict['model']

two_hand_model_dict = pickle.load(open('./two_hand_model.p', 'rb'))
two_hand_model = two_hand_model_dict['model']

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# Define label dictionaries
labels_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '10'}
one_hand_labels_dict = {0: 'C', 1: 'I', 2: 'L', 3: 'O', 4: 'U', 5: 'V'}
two_hand_labels_dict = {
    0: 'A', 1: 'B', 2: 'D', 3: 'E', 4: 'F', 5: 'G', 6: 'H', 7: 'J',
    8: 'K', 9: 'M', 10: 'N', 11: 'P', 12: 'Q', 13: 'R', 14: 'S',
    15: 'T', 16: 'W', 17: 'X', 18: 'Y', 19: 'Z'
}

# Streamlit setup
st.set_page_config(layout="wide")
st.title('Indian Sign Language and Number Recognition')

option = st.sidebar.selectbox('Select Mode', ('Number Detection', 'Word Recognition', 'Alphabet Detection'))

col1, col2 = st.columns([3, 1])
with col1:
    run = st.checkbox('Run', value=True)
    FRAME_WINDOW = st.image([])

with col2:
    st.title("Output")
    output_text_area = st.empty()

cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

def extract_features(hand_landmarks):
    data_aux = []
    x_ = []
    y_ = []
    for landmark in hand_landmarks.landmark:
        x_.append(landmark.x)
        y_.append(landmark.y)
    for landmark in hand_landmarks.landmark:
        data_aux.append(landmark.x - min(x_))
        data_aux.append(landmark.y - min(y_))
    if len(data_aux) > 42:
        data_aux = data_aux[:42]
    else:
        data_aux.extend([0] * (42 - len(data_aux)))
    return data_aux, x_, y_

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

sequence = []
sentence = []
predictions = []
last_detected_number = None
last_detected_word = None
last_detected_alphabet = None
prediction_buffer = []
buffer_size = 5  

while run:
    ret, frame = cap.read()
    if not ret:
        continue

    if option == 'Number Detection':
        with mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.6) as hands:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                                              mp_drawing_styles.get_default_hand_landmarks_style(), 
                                              mp_drawing_styles.get_default_hand_connections_style())
                    data_aux, x_, y_ = extract_features(hand_landmarks)
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_number = labels_dict[int(prediction[0])]

                    prediction_buffer.append(predicted_number)
                    if len(prediction_buffer) > buffer_size:
                        prediction_buffer.pop(0)

                    if prediction_buffer.count(predicted_number) > buffer_size // 2 and predicted_number != last_detected_number:
                        last_detected_number = predicted_number
                        content = translate.run(predicted_number)
                        tts.run(content)
                        output_text_area.subheader(predicted_number)
                        x1 = int(min(x_) * frame.shape[1]) - 10
                        y1 = int(min(y_) * frame.shape[0]) - 10
                        x2 = int(max(x_) * frame.shape[1]) - 10
                        y2 = int(max(y_) * frame.shape[0]) - 10
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                        cv2.putText(frame, predicted_number, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)

    elif option == 'Word Recognition':
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            if results.left_hand_landmarks or results.right_hand_landmarks:
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]
                if len(sequence) == 30:
                    res = lstm_model.predict(np.expand_dims(sequence, axis=0))[0]
                    index = np.argmax(res)
                    predicted_word = actions[index]
                    if predicted_word != last_detected_word:
                        last_detected_word = predicted_word
                        content = translate.run(predicted_word)
                        tts.run(content)
                        output_text_area.subheader(predicted_word)

    elif option == 'Alphabet Detection':
        with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=2) as hands:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                num_hands = len(results.multi_hand_landmarks)
                for hand_landmarks in results.multi_hand_landmarks:
                    data_aux, x_, y_ = extract_features(hand_landmarks)
                    if num_hands == 1:
                        prediction = one_hand_model.predict([np.asarray(data_aux)])
                        predicted_alphabet = one_hand_labels_dict[int(prediction[0])]
                    elif num_hands == 2:
                        prediction = two_hand_model.predict([np.asarray(data_aux)])
                        predicted_alphabet = two_hand_labels_dict[int(prediction[0])]
                    else:
                        predicted_alphabet = 'Unknown'

                    if predicted_alphabet != last_detected_alphabet:
                        last_detected_alphabet = predicted_alphabet
                        content = translate.run(predicted_alphabet)
                        tts.run(content)
                        output_text_area.subheader(predicted_alphabet)
                        x1 = int(min(x_) * frame.shape[1]) - 10
                        y1 = int(min(y_) * frame.shape[0]) - 10
                        x2 = int(max(x_) * frame.shape[1]) - 10
                        y2 = int(max(y_) * frame.shape[0]) - 10
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                        cv2.putText(frame, predicted_alphabet, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)

    FRAME_WINDOW.image(frame)
