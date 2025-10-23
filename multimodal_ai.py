"""
MATCHPLUS Multimodal Emotion AI
Combines body language (pose), facial expression, and vocal tone
for real-time emotional inference.
"""

import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import librosa
import queue
import threading
import time

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Audio queue
audio_q = queue.Queue()

def audio_stream(samplerate=16000, chunk_size=1024):
    """Continuously capture audio and put into queue."""
    def callback(indata, frames, time_, status):
        if status:
            print(status)
        audio_q.put(indata.copy())
    with sd.InputStream(channels=1, samplerate=samplerate, blocksize=chunk_size, callback=callback):
        threading.Event().wait()  # block forever

def extract_audio_features(y, sr):
    """Extract pitch, energy, spectral centroid."""
    y = y.flatten()
    if len(y) < sr // 2:  # minimum 0.5 s
        return np.zeros(3)
    y = y / np.max(np.abs(y) + 1e-6)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitches = pitches[pitches > 0]
    avg_pitch = np.median(pitches) if len(pitches) > 0 else 0
    energy = np.mean(y ** 2)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    return np.array([avg_pitch, energy, centroid])

def classify_tone(features):
    pitch, energy, centroid = features
    if energy < 0.0005:
        return "silent"
    if pitch < 120 and centroid < 1500:
        return "calm"
    elif pitch > 180 and centroid > 2500:
        return "stressed"
    else:
        return "neutral"

# Start audio thread
threading.Thread(target=audio_stream, daemon=True).start()

# Main loop
cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose, \
     mp_face.FaceMesh(min_detection_confidence=0.6, min_tracking_confidence=0.6) as face:

    audio_buffer = []
    tone_state = "silent"
    last_tone_update = time.time()
    update_interval = 10  # seconds

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe processing
        pose_results = pose.process(rgb)
        face_results = face.process(rgb)

        # Pose classification
        body_state = "neutral"
        if pose_results.pose_landmarks:
            left_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            shoulder_dist = abs(left_shoulder.x - right_shoulder.x)
            if shoulder_dist > 0.25:
                body_state = "open_confident"
            elif shoulder_dist < 0.18:
                body_state = "closed_nervous"

        # Face classification
        face_state = "neutral"
        if face_results.multi_face_landmarks:
            lm = face_results.multi_face_landmarks[0].landmark
            mouth_open = abs(lm[13].y - lm[14].y)
            if mouth_open > 0.002:
                face_state = "happy"

        # Audio accumulation
        while not audio_q.empty():
            audio_buffer.append(audio_q.get())

        # Update tone every 10 seconds
        current_time = time.time()
        if current_time - last_tone_update >= update_interval and audio_buffer:
            y = np.concatenate(audio_buffer, axis=0).flatten()
            features = extract_audio_features(y, 16000)
            tone_state = classify_tone(features)
            audio_buffer = []  # reset buffer
            last_tone_update = current_time

        # Fuse emotions
        score = 0
        if face_state == "happy": score += 1
        if body_state == "open_confident": score += 1
        if tone_state == "calm": score += 1
        if tone_state == "stressed": score -= 1
        if body_state == "closed_nervous": score -= 1
        overall = "Neutral / Observing"
        if score >= 2:
            overall = "Positive / Engaged"
        elif score <= -1:
            overall = "Negative / Stressed"

        # Draw
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        if face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(frame, face_results.multi_face_landmarks[0],
                                      mp_face.FACEMESH_CONTOURS)

        color = (0, 255, 0) if "Positive" in overall else ((0, 255, 255) if "Neutral" in overall else (0, 0, 255))
        cv2.putText(frame, f"Face: {face_state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"Body: {body_state}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"Tone: {tone_state}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"Overall: {overall}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        cv2.imshow("MATCHPLUS Multimodal AI", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
