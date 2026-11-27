import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

DATA_PATH = "data/train_and_test2.csv"
FEATURES = ["Pclass", "Age", "Sex", "Fare"]
TARGET = "2urvived"

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="wide")

@st.cache_resource
def train_model():
    data = pd.read_csv(DATA_PATH)

    # Keep only the four features specified in the project statement and the target.
    data = data[FEATURES + [TARGET]].copy()
    data = data.dropna(subset=[TARGET])
    for col in FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        data[col] = data[col].fillna(data[col].median())

    X = data[FEATURES]
    y = data[TARGET].astype(int)

    # Hold-out test set for an honest evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic_regression", LogisticRegression(max_iter=1000, random_state=42))
    ])
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "confusion": confusion_matrix(y_test, predictions),
        "report": classification_report(y_test, predictions, output_dict=True)
    }

    # Refit on all labelled rows for the interactive demo after evaluation.
    model.fit(X, y)
    return model, metrics, len(data), y.value_counts().to_dict()

model, metrics, row_count, label_counts = train_model()

st.title("🚢 Titanic Survival Probability Simulator")
st.write(
    "Choose a passenger's class, age, gender and fare. "
    "The Logistic Regression model estimates the probability of survival."
)
st.caption(
    "Educational demonstration only. This model learns patterns in historical data; "
    "it cannot determine what would have happened to a particular person."
)

left, right = st.columns([1, 1.25])

with left:
    st.subheader("Passenger details")
    with st.form("passenger_form"):
        pclass = st.selectbox("Passenger class", [1, 2, 3], index=2,
                              format_func=lambda x: {1: "1st Class", 2: "2nd Class", 3: "3rd Class"}[x])
        age = st.slider("Age", min_value=0.0, max_value=80.0, value=25.0, step=0.5)
        gender_label = st.selectbox("Gender", ["Female", "Male"])
        fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=25.0, step=1.0)
        submitted = st.form_submit_button("Predict survival probability", type="primary")

    st.info("Dataset coding used: Sex = 0 for Male and Sex = 1 for Female.")

with right:
    st.subheader("Prediction")
    if submitted:
        sex_value = 1 if gender_label == "Female" else 0
        passenger = pd.DataFrame(
            [[pclass, age, sex_value, fare]], columns=FEATURES
        )
        probability = float(model.predict_proba(passenger)[0, 1])
        predicted_class = int(probability >= 0.5)

        st.metric("Estimated survival probability", f"{probability * 100:.1f}%")
        if predicted_class == 1:
            st.success("Model prediction: More likely to survive (probability ≥ 50%).")
        else:
            st.warning("Model prediction: Less likely to survive (probability < 50%).")

        # Local feature contribution: standardized feature value × model coefficient.
        scaler = model.named_steps["scaler"]
        classifier = model.named_steps["logistic_regression"]
        scaled_values = (passenger[FEATURES].iloc[0].to_numpy() - scaler.mean_) / scaler.scale_
        contributions = scaled_values * classifier.coef_[0]

        fig, ax = plt.subplots()
        colors = ["#2E8B57" if value >= 0 else "#C44E52" for value in contributions]
        ax.barh(FEATURES, contributions, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Contribution to model log-odds (relative)")
        ax.set_title("How these inputs influence this prediction")
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "Positive bars push the model toward survival; negative bars push it away. "
            "These are model contributions, not proof of cause or personal destiny."
        )
    else:
        st.write("Enter passenger details and select **Predict survival probability**.")

st.divider()
st.subheader("Model evaluation")
m1, m2, m3 = st.columns(3)
m1.metric("Rows used", f"{row_count:,}")
m2.metric("Hold-out accuracy", f"{metrics['accuracy'] * 100:.1f}%")
m3.metric("Hold-out ROC-AUC", f"{metrics['roc_auc']:.3f}")

st.write("Confusion matrix (actual rows × predicted columns; order: 0 = did not survive, 1 = survived)")
st.dataframe(
    pd.DataFrame(metrics["confusion"],
                 index=["Actual: did not survive", "Actual: survived"],
                 columns=["Predicted: did not survive", "Predicted: survived"]),
    use_container_width=True
)

st.subheader("Global feature impact")
classifier = model.named_steps["logistic_regression"]
impact = pd.DataFrame({
    "Feature": FEATURES,
    "Coefficient": classifier.coef_[0]
}).sort_values("Coefficient")
fig, ax = plt.subplots()
ax.barh(impact["Feature"], impact["Coefficient"])
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Logistic Regression coefficient (standardized inputs)")
ax.set_title("Direction and strength of model relationships")
st.pyplot(fig)
plt.close(fig)
st.caption(
    "A positive coefficient increases predicted survival odds as that feature increases; "
    "a negative coefficient decreases them, holding other inputs constant. "
    "Coefficient size is not a causal effect."
)

with st.expander("About the dataset and limitations"):
    st.write(
        f"The supplied file contains {row_count:,} labelled passenger records. "
        "The project uses Pclass, Age, Sex and Fare, as requested. "
        "A stratified 80/20 split is used for evaluation, then the model is refitted "
        "on all labelled rows for interactive predictions."
    )
    st.write(
        "The supplied file appears to combine records in one CSV and contains the "
        "target column named `2urvived`. Confirm with your instructor that this is "
        "the intended dataset and split protocol. Historical data may contain bias "
        "and missing context; predictions are estimates, not certainties."
    )
