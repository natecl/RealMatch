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

# Setup MediaPipe
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Audio capture queue
audio_q = queue.Queue()

def audio_stream(duration=0.5, samplerate=16000):
    """Continuously record short audio segments into a queue."""
    def callback(indata, frames, time_, status):
        audio_q.put(indata.copy())
    with sd.InputStream(channels=1, samplerate=samplerate, callback=callback):
        while True:
            sd.sleep(int(duration * 1000))

def extract_audio_features(y, sr):
    """Extract simple tone features: pitch, energy, spectral centroid."""
    if len(y) < sr // 4:
        return np.zeros(3)
    y = y.flatten()
    pitch, mag = librosa.piptrack(y=y, sr=sr)
    avg_pitch = np.mean(pitch[pitch > 0]) if np.any(pitch > 0) else 0
    energy = np.mean(y ** 2)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    return np.array([avg_pitch, energy, centroid])

def classify_tone(features):
    """Heuristic: classify vocal tone."""
    pitch, energy, centroid = features
    if energy < 0.001:
        return "silent"
    if pitch < 120 and centroid < 1500:
        return "calm"
    elif pitch > 180 and centroid > 2500:
        return "stressed"
    else:
        return "neutral"

def classify_expression(landmarks):
    """Simple rule-based facial emotion classifier."""
    if landmarks is None:
        return "neutral"
    mouth_open = np.linalg.norm(
        np.array([landmarks[13].y, landmarks[14].y])
    )
    if mouth_open > 0.002:
        return "happy"
    return "neutral"

def classify_posture(landmarks):
    """Roughly classify openness / confidence from shoulder distance."""
    if not landmarks:
        return "neutral"
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    shoulder_dist = abs(left_shoulder.x - right_shoulder.x)
    if shoulder_dist > 0.25:
        return "open_confident"
    elif shoulder_dist < 0.18:
        return "closed_nervous"
    return "neutral"

def fuse_emotions(face, body, tone):
    """Combine all modalities into one label."""
    score = 0
    if face == "happy": score += 1
    if body == "open_confident": score += 1
    if tone == "calm": score += 1
    if tone == "stressed": score -= 1
    if body == "closed_nervous": score -= 1
    if face == "neutral": score += 0

    if score >= 2:
        return "Positive / Engaged"
    elif score <= -1:
        return "Negative / Stressed"
    else:
        return "Neutral / Observing"

def main():
    # Start audio listener in a background thread
    threading.Thread(target=audio_stream, daemon=True).start()

    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose, \
         mp_face.FaceMesh(min_detection_confidence=0.6, min_tracking_confidence=0.6) as face:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pose_results = pose.process(rgb)
            face_results = face.process(rgb)

            # Pose
            body_state = classify_posture(
                pose_results.pose_landmarks.landmark if pose_results.pose_landmarks else None
            )

            # Face
            face_state = classify_expression(
                face_results.multi_face_landmarks[0].landmark if face_results.multi_face_landmarks else None
            )

            # Audio
            tone_state = "silent"
            if not audio_q.empty():
                audio_chunk = audio_q.get()
                y = audio_chunk.flatten()
                features = extract_audio_features(y, 16000)
                tone_state = classify_tone(features)

            # Fuse
            overall = fuse_emotions(face_state, body_state, tone_state)

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

            cv2.imshow('MATCHPLUS Multimodal AI', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("🎙️ Starting MATCHPLUS Multimodal Emotion AI...")
    time.sleep(1)
    main()
