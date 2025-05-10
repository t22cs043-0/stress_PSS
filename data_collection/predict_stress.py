import pandas as pd
import joblib

# 学習済みモデルとスケーラーの読み込み
model = joblib.load("stress_model.pkl")
scaler = joblib.load("feature_scaler.pkl")

# 特徴量データの読み込み（新しい被験者の特徴量ログなど）
df = pd.read_csv("person3_features.csv")

# 必要な列だけ使う（frameは除外）
X = df.drop(columns=["frame"])
X_scaled = scaler.transform(X)

# モデルでストレススコアを予測
predicted_scores = model.predict(X_scaled)

# 結果の出力（例：最終的に平均スコアを出力）
average_stress = predicted_scores.mean()
print(f"推定されたストレススコア（平均）: {average_stress:.2f}")

# 必要ならCSVで保存
df["predicted_stress"] = predicted_scores
df.to_csv("person1_predicted.csv", index=False)
