# object_detection_tracking.py
import cv2
from ultralytics import YOLO

# Load YOLOv8 pretrained model
model = YOLO("yolov8n.pt")  # small model for fast performance

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects
    results = model(frame)

    # Draw bounding boxes
    for r in results:
        boxes = r.boxes.xyxy
        confidences = r.boxes.conf
        class_ids = r.boxes.cls

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[int(class_ids[i])]
            conf = confidences[i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
