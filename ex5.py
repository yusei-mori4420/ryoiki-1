from ultralytics import YOLO
import cv2

VIDEO_PATH = "ex5-26.mp4"
MODEL_PATH = "best260408.pt"
model = YOLO(MODEL_PATH)


cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("動画を開けません")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    results = model(frame, verbose=False)

    helmet_count = 0

    for result in results:
        for box in result.boxes:

            conf = float(box.conf[0])

            helmet_count += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),  
                2
            )


    if frame_count % 30 == 0:
        print(f"{frame_count}フレーム目 : 検出したヘルメット数 : {helmet_count}")

    cv2.imshow("Helmet Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()