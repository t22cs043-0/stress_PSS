import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import numpy as np

# データ読み込み
df = pd.read_csv("labeled_stress_data.csv")

# 特徴量とラベルを分ける（frameは使わない）
X = df.drop(columns=["frame", "stress_score"]).values
y = df["stress_score"].values

# 被験者1の行数（886行）
n_person1 = 886

# 被験者1と被験者2のデータに分割
X_person1 = X[:n_person1]
y_person1 = y[:n_person1]

X_person2 = X[n_person1:]
y_person2 = y[n_person1:]

# スケーラーは被験者1のデータでfit
scaler = StandardScaler()
X_person1_scaled = scaler.fit_transform(X_person1)
X_person2_scaled = scaler.transform(X_person2)

# 例えば被験者1のデータを学習用、被験者2のデータをテスト用に使う
X_train, y_train = X_person1_scaled, y_person1
X_test, y_test = X_person2_scaled, y_person2

# モデル定義
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Support Vector Regression": SVR()
}


# 学習と評価
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


print("y_train unique values:", np.unique(y_train))
print("y_test unique values:", np.unique(y_test))

print("y_test sample:", y_test[:10])
print("y_pred sample:", y_pred[:10])

print("Are there NaNs in y_test?", np.isnan(y_test).any())
print("Are there NaNs in y_pred?", np.isnan(y_pred).any())
print("Are there Inf in y_test?", np.isinf(y_test).any())
print("Are there Inf in y_pred?", np.isinf(y_pred).any())
# 最良モデル（例: Random Forest）を保存
joblib.dump(models["Random Forest"], "stress_model.pkl")
joblib.dump(scaler, "feature_scaler.pkl")
