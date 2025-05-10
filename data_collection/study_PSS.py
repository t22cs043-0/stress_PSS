import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# データ読み込み
df = pd.read_csv("labeled_stress_data.csv")

# 特徴量とラベルに分ける
X = df.drop(columns=["frame", "stress_score"])
y = df["stress_score"]

# 標準化（スケーリング）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 学習用とテスト用に分割
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# モデル定義
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Support Vector Regression": SVR()
}

# モデルごとに学習・予測・評価
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    print(f"モデル: {name}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  MSE  : {mse:.4f}")
    print(f"  R2   : {r2:.4f}")
    print("-" * 40)

# 最も良かったモデル（例：Random Forest）を保存
import joblib
joblib.dump(models["Random Forest"], "stress_model.pkl")
joblib.dump(scaler, "feature_scaler.pkl")
