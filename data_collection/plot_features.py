import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルを読み込む
df = pd.read_csv('facial_features_log.csv')

# フレーム数をX軸に、特徴量をY軸にしてグラフ化
plt.figure(figsize=(12, 6))

# 例：4つの特徴量をそれぞれ描画
plt.plot(df['frame'], df['right_eye_brow'], label='Right Eye-Brow Distance')
plt.plot(df['frame'], df['left_eye_brow'], label='Left Eye-Brow Distance')
plt.plot(df['frame'], df['mouth_opening'], label='Mouth Opening')
plt.plot(df['frame'], df['cheek_asymmetry'], label='Cheek Asymmetry')

# グラフの装飾
plt.title('Facial Feature Changes Over Time')
plt.xlabel('Frame')
plt.ylabel('Normalized Distance / Value')
plt.legend()
plt.grid(True)

# グラフ表示
plt.tight_layout()
plt.show()
