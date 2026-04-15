## Customer Churn ANN

This project trains a simple artificial neural network for customer churn prediction and serves it with a Streamlit app.

## Project Files

- [model.ipynb](model.ipynb) trains the model, evaluates it, and saves the artifacts.
- [app.py](app.py) is the Streamlit inference app.
- [Artificial_Neural_Network_Case_Study_data.csv](Artificial_Neural_Network_Case_Study_data.csv) is the training dataset.
- [artifacts/](artifacts/) stores the saved model and preprocessing files.

## Requirements

Use Python 3.13 with the project virtual environment, then install dependencies.

```bash
uv sync
```

If you prefer pip, install from `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Train the Model

1. Open [model.ipynb](model.ipynb).
2. Run all cells from top to bottom.
3. The last cell saves these files into [artifacts/](artifacts/):
   - `churn_ann_model.keras`
   - `scaler.pkl`
   - `feature_columns.json`

The app depends on those saved files, so the notebook must be run at least once before launching Streamlit.

## Run the App

```bash
streamlit run app.py
```

Enter customer details in the form and click Predict Churn to see both churn probability and stay probability.

## Notes

- The model uses the same preprocessing in both the notebook and the app: gender encoding, geography one-hot encoding, and standard scaling.
- A churn probability below 50% is shown as `WILL STAY`.
- If Streamlit shows a missing artifact error, re-run [model.ipynb](model.ipynb) to regenerate the files in [artifacts/](artifacts/).
