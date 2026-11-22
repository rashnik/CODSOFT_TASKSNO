"""
CODSOFT TASK 5 — FACE DETECTION
Install:
    pip install opencv-python

Image mode:
    python Task5_Face_Detection.py --image photo.jpg

Webcam mode:
    python Task5_Face_Detection.py --webcam

This project detects faces; it does not identify people.
"""
import argparse
from pathlib import Path
import cv2

def load_detector():
    cascade_file = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_file))
    if detector.empty():
        raise RuntimeError("Could not load Haar cascade classifier.")
    return detector

def detect_faces(frame, detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 220, 0), 2)
        cv2.putText(frame, "Face", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
    return frame, len(faces)

def image_mode(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to open image: {image_path}")
    result, count = detect_faces(image, load_detector())
    output_path = str(Path(image_path).with_name(Path(image_path).stem + "_detected.jpg"))
    cv2.imwrite(output_path, result)
    print(f"Detected {count} face(s). Result saved to: {output_path}")

def webcam_mode():
    detector = load_detector()
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Unable to access webcam. Check camera permissions.")
    print("Webcam started. Press q to quit.")
    while True:
        success, frame = camera.read()
        if not success:
            break
        result, count = detect_faces(frame, detector)
        cv2.putText(result, f"Faces: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)
        cv2.imshow("CODSOFT Face Detection", result)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face detection with OpenCV.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to an image")
    group.add_argument("--webcam", action="store_true", help="Detect faces from webcam")
    args = parser.parse_args()
    if args.image:
        image_mode(args.image)
    else:
        webcam_mode()
