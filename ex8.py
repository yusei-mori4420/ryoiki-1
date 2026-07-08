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
stationary_count = 0         
snap_frames = []   

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    results = model.track(frame, persist=True, verbose=False)

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

    
    if moving_count >= 3:
        team_state = "TEAM MOVING"
    else:
        team_state = "TEAM STATIONARY"

    
    cv2.putText(
        frame,
        team_state,
        (frame.shape[1] - 330, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )

    
    if team_state == "TEAM STATIONARY":
        stationary_count += 1
    else:
        if stationary_count >= 30:
            print(f"スナップしたフレーム: {frame_count}")
            snap_frames.append(frame_count)

        stationary_count = 0

    cv2.imshow("Helmet Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print("スナップしたフレーム一覧")
print(snap_frames)

cap.release()
cv2.destroyAllWindows()