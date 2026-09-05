# Message Guard

A Flask frontend for the spam classifier in `spam.ipynb`.

## Setup

Place the SMS dataset at `data/spam.csv`. It must contain the original columns `v1` (label: `spam` or `ham`) and `v2` (message text).

```powershell
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000 in a browser, paste a message, and select **Check message**.

The app trains the TF-IDF and Linear SVC pipeline from `spam.csv` when it starts. The notebook's prediction bug is avoided by keeping preprocessing, vectorization, and classification in the same pipeline.
