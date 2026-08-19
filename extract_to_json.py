import argparse
import json
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp


# 33 點骨架連線 (MediaPipe Pose 定義)
POSE_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    (27, 31),
    (28, 32),
]


def parse_args():
    """解析命令列參數。"""
    parser = argparse.ArgumentParser(
        description="從影片擷取 MediaPipe Pose 33 點 3D 資料並輸出階層 JSON。"
    )
    parser.add_argument(
        "-i",
        "--input_video",
        type=str,
        required=True,
        help="輸入影片路徑，例如：C:/path/to/video.mp4",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./pose_landmarker_heavy.task",
        help="Pose Landmarker 模型路徑 (.task)。若不存在會自動下載。",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./DATASET",
        help="輸出資料夾，預設為 ./DATASET",
    )
    return parser.parse_args()


def ensure_pose_landmarker_model(model_path: Path):
    """確保 Pose Landmarker 模型檔存在，不存在就下載。"""
    if model_path.exists():
        return

    model_url = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    )
    print(f"找不到模型檔，開始下載：{model_url}")
    urlretrieve(model_url, str(model_path))
    print(f"模型下載完成：{model_path}")


def draw_pose_landmarks(frame_bgr, normalized_landmarks):
    """繪製 33 點骨架，方便輸出可視化影片。"""
    h, w = frame_bgr.shape[:2]
    points = []

    for lm in normalized_landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((x, y, lm.visibility))

    for start_idx, end_idx in POSE_CONNECTIONS:
        x1, y1, v1 = points[start_idx]
        x2, y2, v2 = points[end_idx]
        if v1 > 0.1 and v2 > 0.1:
            cv2.line(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for x, y, vis in points:
        if vis > 0.1:
            cv2.circle(frame_bgr, (x, y), 3, (0, 0, 255), -1)


def build_frame_landmark_list(world_landmarks, normalized_landmarks):
    """
    產生單幀資料：
    - 一個 list，內含 33 個 dict
    - 每個 dict 格式為 {x, y, z, visibility}
    - 優先使用 world_landmarks，若無則 fallback 到 normalized_landmarks
    """
    one_frame = []

    for idx in range(33):
        # 優先 world；若不存在則 fallback 到 normalized
        source_lm = world_landmarks[idx] if world_landmarks is not None else normalized_landmarks[idx]
        one_frame.append(
            {
                "x": float(source_lm.x),
                "y": float(source_lm.y),
                "z": float(source_lm.z),
                "visibility": float(source_lm.visibility),
            }
        )

    return one_frame


def main():
    args = parse_args()
    input_path = Path(args.input_video)
    model_path = Path(args.model_path)
    dataset_dir = Path(args.dataset_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"找不到輸入影片：{input_path}")

    dataset_dir.mkdir(parents=True, exist_ok=True)
    ensure_pose_landmarker_model(model_path)

    # 輸出固定放進 DATASET 目錄
    output_json_path = dataset_dir / f"{input_path.stem}_3d_data.json"
    output_video_path = dataset_dir / f"{input_path.stem}_output.mp4"

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片：{input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"無法建立輸出影片：{output_video_path}")

    # 外層 list：整段影片時間序列
    all_frames_data = []
    frame_index = 0

    # mediapipe.tasks Pose Landmarker 初始化
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(round((frame_index / fps) * 1000.0))

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # 只保留成功偵測影格；每幀必須剛好 33 個點
            if result.pose_landmarks:
                normalized_landmarks = result.pose_landmarks[0]
                world_landmarks = result.pose_world_landmarks[0] if result.pose_world_landmarks else None

                one_frame = build_frame_landmark_list(
                    world_landmarks=world_landmarks,
                    normalized_landmarks=normalized_landmarks,
                )
                all_frames_data.append(one_frame)

                draw_pose_landmarks(frame_bgr, normalized_landmarks)

            writer.write(frame_bgr)
            frame_index += 1

    cap.release()
    writer.release()

    # 直接輸出為 JSON 陣列（UTF-8）
    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(all_frames_data, f, ensure_ascii=False)

    print(f"處理完成：{input_path}")
    print(f"輸出影片：{output_video_path}")
    print(f"輸出 JSON：{output_json_path}")
    print(f"共處理影格數：{frame_index}")
    print(f"成功偵測並寫入 JSON 的影格數：{len(all_frames_data)}")


if __name__ == "__main__":
    main()

