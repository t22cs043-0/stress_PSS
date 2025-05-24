import pandas as pd

# Person 1 のデータ読み込みとラベル追加
df1 = pd.read_csv("person1_features.csv")#komorikarin
original_score=39
normalized_score = original_score / 56.0 * 10.0#正規化
df1["stress_score"] = normalized_score  # Person 1 のPSSスコア（スケーリング済み）

# Person 2 のデータ
df2 = pd.read_csv("person2_features.csv")#kawamurashungo
original_score=38
normalized_score = original_score / 56.0 * 10.0
df2["stress_score"] = normalized_score  # Person 2 のPSSスコア（スケーリング済み）

# 結合して1つの学習データに
merged = pd.concat([df1, df2], ignore_index=True)

# 保存
merged.to_csv("labeled_stress_data.csv", index=False)