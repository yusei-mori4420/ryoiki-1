from ultralytics import YOLO
import cv2
import torch

model = YOLO("yolov8n.pt")

img_path = r"C:\Users\shazm\ryoiki\git\ex1-26.png"

results = model(img_path, conf=0.4)

img = results[0].orig_img
boxes = results[0].boxes

person_count = 0

for box in boxes:
    cls = int(box.cls[0])  

    if cls == 0:
        person_count += 1

        x1, y1, x2, y2 = box.xyxy[0]

        cv2.rectangle(
            img,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 0, 255), 
            3
        )

print("人数:", person_count)

cv2.imshow("", img)
cv2.waitKey(0)
cv2.destroyAllWindows()