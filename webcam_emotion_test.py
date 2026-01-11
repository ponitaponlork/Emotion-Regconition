"""
Webcam-based tester for a six-class emotion detection model.

Requirements:
    pip install tensorflow opencv-python numpy

The script assumes the Keras/TensorFlow model expects grayscale 48x48 inputs
with pixel values in [0, 1]. Update EMOTIONS or preprocessing if your training
pipeline differs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model


EMOTIONS = (
    "angry",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral"
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run emotion detection on webcam feed."
    )
    parser.add_argument(–
        "--model",
        type=Path,
        default=Path(__file__).with_name("emotion_model_best.h5"),
        help="Path to the trained .h5 model file.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Webcam index to open (default: 0).",
    )
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=100,
        help="Minimum face size (pixels) for detection.",
    )
    return parser.parse_args()


def preprocess(face: np.ndarray) -> np.ndarray:
    """Resize, normalize, and add batch/channel dims."""
    face_resized = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)
    face_resized = face_resized.astype("float32") / 255.0
    face_resized = np.expand_dims(face_resized, axis=(0, -1))
    return face_resized


def main() -> None:
    args = parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")

    print(f"Loading model from {args.model}")
    model = load_model(args.model)

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(str(cascade_path))
    if face_detector.empty():
        raise RuntimeError("Failed to load Haar cascade for face detection.")

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Check camera permissions/index.")

    print("Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(args.min_face_size, args.min_face_size),
            )

            for (x, y, w, h) in faces:
                roi_gray = gray[y : y + h, x : x + w]
                input_tensor = preprocess(roi_gray)
                preds = model.predict(input_tensor, verbose=0)[0]
                emotion_idx = int(np.argmax(preds))
                emotion = EMOTIONS[emotion_idx]
                confidence = float(preds[emotion_idx])

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
                text = f"{emotion} ({confidence:.2f})"
                cv2.putText(
                    frame,
                    text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Emotion Detector", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

