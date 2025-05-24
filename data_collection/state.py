import os,sys # システムに関するモジュール
import mediapipe as mp # ランドマーク描画に関するモジュール
import numpy as np # 配列に関するモジュール
import pandas as pd # データ解析に関するモジュール
from tkinter import filedialog # ファイルダイアログにおけるライブラリ
import cv2 # 画像描画に関するモジュール
import time,datetime # 時刻に関するモジュール

"ハイパーパラメータ"
target_id = 1 # ターゲットID
word_id = 1 # 単語ID(0:Hello , 1:Nice to me too , 2:Excuse me)

filename = "facemesh_"+str(target_id)+"_word_"+str(target_id) # 保存ファイル名
landmark_size = 478 # ランドマークサイズ
columns = ["face_"+str(i)+"_"+str(j) for i in range(landmark_size) for j in range(3)] # 列名
columns = ["time"]+columns # time列追加
data = None # 保存データ
df = None # データフレーム
mp_drawing = mp.solutions.drawing_utils # 描画用インスタンス
mp_face_mesh = mp.solutions.face_mesh # Facemeshインスタンス
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True,min_detection_confidence=0.5, min_tracking_confidence=0.5)
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1) # Facemesh描画設定
cap = cv2.VideoCapture(0) # ビデオキャプチャ
fps = 25 # 1秒当たりのフレーム数
spf = 1.0/fps # 1フレーム当たりの秒数
frame_fps = None # 暫定fps
capture_flag = False # キャプチャフラグ
capture_start_time = None # キャプチャ開示時刻
face_color = (255,0,0) # Fasemeshの画素
count_enter = 0 # Enterカウント

while cap.isOpened(): # キャプチャが終わるまで繰り返し処理
    frame_time_start = time.time() # 描画開始時刻
    tick = cv2.getTickCount() #
    _,frame = cap.read() # フレーム取得
    height,width,_ = frame.shape # フレームサイズ取得
    #print(height,",",width) # デバッグ用
    if not _: # フレーム取得失敗時
        print("動画フレームの取得に失敗しました．") # エラー
        break # ブレイク

    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB) # BGRをRGB色空間に変換
    results = face_mesh.process(frame) # フレームからFacemesh取得
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # 色空間をBGRに戻す
    cv2.rectangle(frame,pt1=(0,0),pt2=(width,height),color=(0,0,0),thickness=-1)
    
    if results.multi_face_landmarks: # Fasemeshに結果が反映されているとき
        for landmarks in results.multi_face_landmarks:
            #print(landmarks.landmark) # デバッグ用
            landmarks_position = np.ndarray((0,3)) # ランドマーク用配列
            for idx, landmark in enumerate(landmarks.landmark):
                x = landmark.x; y = landmark.y; z = landmark.z # 各座標取得
                landmarks_position = np.append(landmarks_position,np.array([[x,y,z]]),axis=0) # ランドマークを配列へ追加
                cv2.circle(frame, (int(x*width),int(y*height)),2,face_color,-1) # 点描画
            #print(landmarks_position.shape) # デバッグ用
            #mp_drawing.draw_landmarks(image=frame,landmark_list=landmarks,landmark_drawing_spec=drawing_spec,connection_drawing_spec=drawing_spec) # ランドマーク描画
    if frame_fps is not None: cv2.putText(frame, "FPS: "+str(frame_fps),(width-80,20),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1,cv2.LINE_AA) # 文字表示
    cv2.putText(frame, "Enter count: "+str(count_enter),(10,25),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2,cv2.LINE_AA) # 文字表示
    cv2.imshow('FaceMesh ('+filename+")", frame) # フレーム描画

    key = cv2.waitKey(1) # 押下キー取得
    if key&0xFF==27: # esc
        break # ブレイク
    elif key&0xFF==32: # space
        #print("space ok",key) # デバッグ用
        if capture_flag: # キャプチャフラグが立っているとき
            capture_flag = False # キャプチャフラグを降ろす
            print("recording end.\n") # デバッグ用
            print(data.shape) # デバッグ用
            face_color = (255,0,0) # Fasemesh色画素変更
        else: # キャプチャフラグが降りているとき
            capture_flag = True # キャプチャフラグを立てる
            data = None # 入力データリセット
            capture_start_time = time.time() # キャプチャ開始時刻
            print("recording start.") # デバッグ用
            face_color = (0,0,255) # Fasemesh色画素変更
    elif key&0xFF==13: # Enter
        count_enter += 1 # Enterカウント増加
    elif key&0xFF==8: # Backspace
        count_enter = 0 # Enterカウントリセット
    elif key&0xFF==115: # s
        if data is not None: # データが存在するとき
            df = pd.DataFrame(data,columns=columns) # データフレーム作成
            save_dir = "../raw_data/" # 保存先ディレクトリ
            if not os.path.exists(save_dir): # 指定したパスが存在しないとき
                os.makedirs(save_dir) # ディレクトリ作成
            file_path = filedialog.asksaveasfilename(initialdir=save_dir,initialfile=filename,title="名前を付けて保存",filetypes=[("CSV",".csv")]) # ファイル選択したときのパス取得
            if file_path!="": # ファイル名が入力されている場合
                if file_path[-4:]!=".csv": file_path += ".csv" # 拡張子を付ける
                df.to_csv(file_path,index=None) # csvファイルで保存
                print("save completed.") # デバッグ用
                break # ブレイク
        else: print("data is not found.") # データがないときエラー
            
    if capture_flag: # キャプチャフラグが立っているとき
        data_tmp = np.insert(landmarks_position,0,time.time()-capture_start_time) # タイムスタンプ追加(478,3)+1→(1435)
        data_tmp = data_tmp.reshape(1,-1) # (1435)→(1,1435)
        if data is None: data = data_tmp # データが未定義であるとき，その値とする
        else: data = np.append(data,data_tmp,axis=0) # バッチ方向へ連結(T,D)

    frame_time_draw = time.time()-frame_time_start # フレーム処理終了時刻
    if frame_time_draw<spf: time.sleep(spf-frame_time_draw) # 一定時間休止
    frame_fps = int(1.0/(time.time()-frame_time_start)) # 暫定fps

face_mesh.close() # Fasemeshインスタンスを閉じる
cap.release() # キャプチャインスタンスを閉じる
