## Customer Churn ANN

This project trains a simple ANN churn classifier in `model.ipynb` and serves predictions in `app.py`.

### Run the notebook

1. Open `model.ipynb`.
2. Run all cells from top to bottom.
3. The final cell saves the trained artifacts into `artifacts/`:
	- `churn_ann_model.keras`
	- `scaler.pkl`
	- `feature_columns.json`

### Run the app

```bash
streamlit run app.py
```

If the artifacts are missing, the app will prompt you to run the notebook first.
