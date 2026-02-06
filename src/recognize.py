import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort
import pickle
import os

THRESHOLD = 0.62

detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options.intra_op_num_threads = max(1, (os.cpu_count() or 1) - 1)
session = ort.InferenceSession("../models/embedder_arcface.onnx", sess_options=session_options)

REF_POINTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

INDICES = [33, 263, 1, 61, 291]

def preprocess(aligned):
    img = aligned.astype(np.float32)
    img = (img - 127.5) / 127.5
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

# Load database and precompute mean embeddings
with open('../data/db/face_db.pkl', 'rb') as f:
    db = pickle.load(f)

reference = {}
for name, embs in db.items():
    mean_emb = np.mean(np.array(embs), axis=0)
    mean_emb /= np.linalg.norm(mean_emb)
    reference[name] = mean_emb
reference_names = list(reference.keys())
reference_matrix = np.stack([reference[name] for name in reference_names], axis=0) if reference_names else np.empty((0, 0))

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        crop = frame[y:y+h, x:x+w]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0]
            pts = np.array([[lm.landmark[i].x * w, lm.landmark[i].y * h] for i in INDICES], dtype=np.float32)
            M, _ = cv2.estimateAffinePartial2D(pts, REF_POINTS)
            aligned = cv2.warpAffine(crop, M, (112, 112), flags=cv2.INTER_LINEAR)
            blob = preprocess(aligned)
            query_emb = session.run(None, {'input.1': blob})[0][0]
            query_emb = query_emb / np.linalg.norm(query_emb)

            if reference_matrix.size > 0:
                sims = reference_matrix @ query_emb
                best_index = int(np.argmax(sims))
                max_sim = float(sims[best_index])
                identity = reference_names[best_index]
            else:
                max_sim = -1
                identity = "Unknown"

            label = identity if max_sim >= THRESHOLD else "Unknown"
            color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
            cv2.putText(frame, f"{label} ({max_sim:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.imshow('Aligned', aligned)
    cv2.imshow('Live Recognition', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
