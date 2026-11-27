"""Optional command-line training/evaluation script."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

DATA_PATH = "data/train_and_test2.csv"
FEATURES = ["Pclass", "Age", "Sex", "Fare"]
TARGET = "2urvived"

df = pd.read_csv(DATA_PATH)[FEATURES + [TARGET]].copy()
df = df.dropna(subset=[TARGET])
for col in FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())

X, y = df[FEATURES], df[TARGET].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic_regression", LogisticRegression(max_iter=1000, random_state=42))
])
model.fit(X_train, y_train)
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, prob), 4))
print(classification_report(y_test, pred))
