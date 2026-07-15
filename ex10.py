from ultralytics import YOLO
import torch
import math

MODEL_PATH = "best260408.pt"

MOVE_THRESHOLD = 4        
MOVING_HELMETS = 3       
STATIONARY_FRAMES = 30    


IMGSZ = 416               
VID_STRIDE = 1            
CONF = 0.4               


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():   
        return "mps"
    return "cpu"


DEVICE = pick_device()
print(f"使用デバイス: {DEVICE}")


def find_snap_frames(video_path):
    
    model = YOLO(MODEL_PATH)

    prev_centers = {}
    stationary_count = 0
    snap_frames = []

    
    results = model.track(
        source=video_path,
        stream=True,
        persist=True,
        verbose=False,
        imgsz=IMGSZ,
        conf=CONF,
        device=DEVICE,
        half=(DEVICE == "cuda"),   
        vid_stride=VID_STRIDE,
    )

    print(f"\n{video_path} の処理を開始します")

    for i, result in enumerate(results):
        
        frame_count = (i + 1) * VID_STRIDE

        if frame_count % 100 < VID_STRIDE:
            print(f"{video_path}: {frame_count}フレーム処理済み")

        moving_count = 0
        current_ids = set()

        if result.boxes is not None and result.boxes.id is not None:
            ids = result.boxes.id.int().cpu().tolist()
            xyxy = result.boxes.xyxy.cpu().tolist()

            for track_id, (x1, y1, x2, y2) in zip(ids, xyxy):
                current_ids.add(track_id)

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                if track_id in prev_centers:
                    prev_cx, prev_cy = prev_centers[track_id]
                    dx = cx - prev_cx
                    dy = cy - prev_cy
                    distance = math.sqrt(dx * dx + dy * dy)

                    
                    if distance >= MOVE_THRESHOLD * VID_STRIDE:
                        moving_count += 1

                prev_centers[track_id] = (cx, cy)


        prev_centers = {
            k: v for k, v in prev_centers.items() if k in current_ids
        }

        team_moving = moving_count >= MOVING_HELMETS

        if not team_moving:
            stationary_count += 1
        else:
            if stationary_count >= STATIONARY_FRAMES // VID_STRIDE:
                snap_frames.append(frame_count)
                print(f"{video_path} スナップ検出: フレーム {frame_count}")
            stationary_count = 0

    print(f"{video_path} の処理が終了しました")
    return snap_frames


snap_frames_ex5 = find_snap_frames("ex5-26.mp4")
snap_frames_ex9 = find_snap_frames("ex9-26.mp4")

print("\n==============================")
print("処理結果")
print("==============================")
print("ex5-26.mp4のスナップフレーム:", snap_frames_ex5)
print("ex9-26.mp4のスナップフレーム:", snap_frames_ex9)

if len(snap_frames_ex5) != len(snap_frames_ex9):
    print("\n注意: 2本の動画で検出されたスナップ数が異なります")
    print(f"ex5: {len(snap_frames_ex5)}個 / ex9: {len(snap_frames_ex9)}個")

number_of_pairs = min(len(snap_frames_ex5), len(snap_frames_ex9))
differences = []

for i in range(number_of_pairs):
    d = abs(snap_frames_ex5[i] - snap_frames_ex9[i])
    differences.append(d)
    print(f"{i + 1}回目の差: |{snap_frames_ex5[i]} - {snap_frames_ex9[i]}| = {d}")

print("\nスナップフレーム番号の差の一覧")
print(differences)