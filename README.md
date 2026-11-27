# Titanic Survival Probability Simulator

## Project statement
Build an interactive app where users enter passenger class, age, gender and fare.
A Logistic Regression model estimates survival probability and displays feature-impact charts.

## Dataset
Place the supplied `train_and_test2.csv` file inside the `data/` folder.
The app expects the target column to be named `2urvived` and uses:
`Pclass`, `Age`, `Sex`, `Fare`.

In this dataset, `Sex=0` means Male and `Sex=1` means Female.
The app reports the dataset coding on screen.

## Run locally
1. Install Python 3.10 or later.
2. Open a terminal in this project folder.
3. Create and activate a virtual environment (optional but recommended).
4. Install packages: `pip install -r requirements.txt`
5. Start the app: `streamlit run app.py`
6. Open the local URL printed by Streamlit (usually http://localhost:8501).

## Deploy online
1. Upload this project to a GitHub repository, including the `data/` folder if your
   instructor permits sharing the dataset.
2. Open Streamlit Community Cloud and sign in with GitHub.
3. Choose **Create app**, select your repository, branch and `app.py`.
4. Deploy and test the public app URL.
5. Do not publish private student details or credentials.

## Evaluation
The app uses a stratified 80/20 train/test split (random_state=42), reports accuracy,
ROC-AUC and a confusion matrix, then refits the model on all labelled rows for the
interactive demo. `python train_model.py` runs a separate evaluation in the terminal.

## Important limitations
The provided CSV has 1,309 rows and a column named `2urvived`. It looks like one
combined labelled dataset rather than separate train and test files. Confirm the
required protocol with your instructor. Model probabilities are estimates, not facts.
Feature contributions describe the fitted model, not causal effects.
