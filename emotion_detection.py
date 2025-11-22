import cv2
import mediapipe as mp
import numpy as np
from collections import deque, Counter
import joblib

# Load trained model + label encoder
model = joblib.load("landmark_emotion_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Smoothing window
SMOOTH_WINDOW = 7
recent_predictions = deque(maxlen=SMOOTH_WINDOW)

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,   # MUST stay False for 468 landmarks
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

face_detector = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.6
)

def smooth_label(queue):
    if not queue:
        return None
    count = Counter(queue)
    return count.most_common(1)[0][0]

cap = cv2.VideoCapture(0)
print("Running... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    detection = face_detector.process(rgb)

    if detection.detections:
        det = detection.detections[0]
        box = det.location_data.relative_bounding_box

        x1 = int(box.xmin * w)
        y1 = int(box.ymin * h)
        x2 = x1 + int(box.width * w)
        y2 = y1 + int(box.height * h)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        face_region = rgb[y1:y2, x1:x2]
        mesh_result = face_mesh.process(face_region)

        if mesh_result.multi_face_landmarks:
            lm = mesh_result.multi_face_landmarks[0].landmark

            # ✅ ALWAYS LIMIT TO FIRST 468 LANDMARKS
            lm = lm[:468]

            landmarks = []
            for p in lm:
                landmarks.extend([p.x, p.y, p.z])

            input_data = np.array(landmarks).reshape(1, -1)

            probs = model.predict_proba(input_data)[0]
            emotion = label_encoder.inverse_transform([np.argmax(probs)])[0]
            confidence = np.max(probs) * 100

            recent_predictions.append(emotion)
            display_emotion = smooth_label(recent_predictions)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{display_emotion} ({confidence:.1f}%)",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)

    else:
        cv2.putText(frame, "Face Not Detected",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 2)

    cv2.imshow("Emotion Detector by Aman:The Genius", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
