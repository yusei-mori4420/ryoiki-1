import cv2
from ultralytics import YOLO
import torch

# YOLOv8モデルをロード
model = YOLO("best260408.pt")

# ビデオファイルを開く
video_path = "ex5-26.mp4"
cap = cv2.VideoCapture(video_path)

# ビデオフレームをループする
while cap.isOpened():
    # ビデオからフレームを読み込む
    success, frame = cap.read()

    if success:
        # フレームでYOLOv8トラッキングを実行し、フレーム間でトラックを永続化
        results = model.track(frame, persist=True)

        # フレームに結果を可視化
        annotated_frame = results[0].plot()

        # 注釈付きのフレームを表示
        cv2.imshow("YOLOv8トラッキング", annotated_frame)

        # 'q'が押されたらループから抜ける
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # ビデオの終わりに到達したらループから抜ける
        break

# ビデオキャプチャオブジェクトを解放し、表示ウィンドウを閉じる
cap.release()
cv2.destroyAllWindows()
