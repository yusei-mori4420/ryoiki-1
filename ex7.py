from ultralytics import YOLO
import cv2
import math

VIDEO_PATH = "ex5-26.mp4"
MODEL_PATH = "best260408.pt"

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("動画を開けません")
    exit()

prev_centers = {}
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    results = model.track(frame, persist=True, verbose=False)

    helmet_count = 0
    moving_count = 0  

    for result in results:
        for box in result.boxes:

            if box.id is None:
                continue

            track_id = int(box.id[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            color = (0, 255, 0)

            if track_id in prev_centers:
                prev_cx, prev_cy = prev_centers[track_id]

                dx = cx - prev_cx
                dy = cy - prev_cy
                distance = math.sqrt(dx * dx + dy * dy)

                if distance >= 4:
                    color = (0, 0, 255)
                    moving_count += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            prev_centers[track_id] = (cx, cy)
            helmet_count += 1

    if moving_count >= 3:
        text = "TEAM MOVING"
    else:
        text = "TEAM STATIONARY"

    cv2.putText(
        frame,
        text,
        (frame.shape[1] - 330, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )

    cv2.imshow("Helmet Movement Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()