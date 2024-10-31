import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the one-hand and two-hand models
one_hand_model_dict = pickle.load(open('./one_hand_model.p', 'rb'))
one_hand_model = one_hand_model_dict['model']

two_hand_model_dict = pickle.load(open('./two_hand_model.p', 'rb'))
two_hand_model = two_hand_model_dict['model']

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe hands solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=2)

# Define the label dictionaries for one-hand and two-hand gestures
one_hand_labels_dict = {0: 'C', 1: 'I', 2: 'L', 3: 'O', 4: 'U', 5: 'V'}
two_hand_labels_dict = {
    0: 'A', 1: 'B', 2: 'D', 3: 'E', 4: 'F', 5: 'G', 6: 'H', 7: 'J',
    8: 'K', 9: 'M', 10: 'N', 11: 'P', 12: 'Q', 13: 'R', 14: 'S',
    15: 'T', 16: 'W', 17: 'X', 18: 'Y', 19: 'Z'
}

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        num_hands_detected = len(results.multi_hand_landmarks)
        model = None
        labels_dict = None

        if num_hands_detected == 1:
            # Use the one-hand model
            model = one_hand_model
            labels_dict = one_hand_labels_dict
        elif num_hands_detected == 2:
            # Use the two-hand model
            model = two_hand_model
            labels_dict = two_hand_labels_dict

        if model:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                min_x = min(x_)
                max_x = max(x_)
                min_y = min(y_)
                max_y = max(y_)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min_x)
                    data_aux.append(y - min_y)

            expected_feature_length = 42 if num_hands_detected == 1 else 84
            if len(data_aux) == expected_feature_length:
                x1 = int(min_x * W) - 20
                y1 = int(min_y * H) - 20
                x2 = int(max_x * W) + 20
                y2 = int(max_y * H) + 20

                prediction = model.predict([np.asarray(data_aux)])

                predicted_character = prediction[0]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 7)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3,
                            cv2.LINE_AA)
            else:
                print(f"Expected {expected_feature_length} features, but got {len(data_aux)} features instead.")

    cv2.imshow('frame', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()