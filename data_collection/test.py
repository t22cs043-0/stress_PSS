import cv2
import mediapipe as mp
import numpy as np
import csv

# 補助関数：距離と角度計算
def calc_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eyebrow_eye_distance(landmarks):
    # 右目上：159，右眉：285、左目上：386、左眉：55
    right = calc_distance(landmarks[159], landmarks[285])
    left = calc_distance(landmarks[386], landmarks[55])
    return right, left

def mouth_opening(landmarks):
    # 上唇：13, 下唇：14
    return calc_distance(landmarks[13], landmarks[14])

def cheek_asymmetry(landmarks):
    # 右頬：234、左頬：454、顔中心：1
    r = calc_distance(landmarks[234], landmarks[1])
    l = calc_distance(landmarks[454], landmarks[1])
    return abs(r - l)

def extract_features(landmarks):
    r_eye_brow, l_eye_brow = eyebrow_eye_distance(landmarks)
    mouth = mouth_opening(landmarks)
    cheeks = cheek_asymmetry(landmarks)
    return {
        "right_eye_brow": r_eye_brow,
        "left_eye_brow": l_eye_brow,
        "mouth_opening": mouth,
        "cheek_asymmetry": cheeks
    }

# MediaPipe 初期化
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# CSV出力の準備
csv_file = open("person3_features.csv", 'w', newline='')
csv_writer = csv.DictWriter(csv_file, fieldnames=[
    "frame", "right_eye_brow", "left_eye_brow", "mouth_opening", "cheek_asymmetry"
])
csv_writer.writeheader()


# カメラ起動
cap = cv2.VideoCapture(0)
frame_count = 0

import time
start_time = time.time()
duration = 30

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    
    # 鏡映像に反転
    frame = cv2.flip(frame, 1)

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 描画と特徴量抽出のための座標取得
            landmarks = []
            for lm in face_landmarks.landmark:
                h, w, _ = frame.shape
                x, y = int(lm.x * w), int(lm.y * h)
                landmarks.append((lm.x, lm.y))  # 特徴量抽出用 (正規化座標)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)  # 表示用

            # 特徴量の抽出とCSV出力
            features = extract_features(landmarks)
            features["frame"] = frame_count
            csv_writer.writerow(features)
            frame_count += 1
    # 時間制限チェック
    elapsed = time.time() - start_time
    if elapsed > duration:
        break

    # 経過秒数を画面に表示
    cv2.putText(
        frame,
        f"Time: {int(elapsed)}s / {duration}s",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.imshow("FaceMesh", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()
