# Salary Predictor 💰 — GUI Edition

A Random Forest model that predicts salary from 9 job-market features, now with a
**black & sea-green GUI** built in Streamlit — rebuilt from the original
command-line tool.

| | |
|---|---|
| **Model** | Random Forest Regressor (≈95.6% R²) |
| **Features** | job title, experience, education, skills count, industry, company size, location, remote work, certifications |
| **GUI** | `app.py` — Streamlit, dark theme, sea-green gradient |
| **Original CLI** | `cli_app.py` — kept for terminal use |

---

## 1. Project structure

```
DS_Project/
├── app.py                 # GUI app — run this (streamlit run app.py)
├── cli_app.py             # original terminal version
├── best_model_rf.pkl      # trained Random Forest model (~190 MB)
├── label_encoders.pkl     # encoders for the 6 categorical features
├── scaler.pkl             # scaler (used for the linear-regression baseline, kept for reference)
├── project.ipynb          # full notebook: EDA → cleaning → modelling
├── requirements.txt
├── .gitignore
└── .gitattributes         # tells Git to use LFS for the .pkl model
```

---

## 2. Setup

You need **Python 3.9+** and `pip`.

```bash
# 1. Move into the project folder
cd DS_Project

# 2. Create and activate a virtual environment (recommended)
python3 -m venv venv

# macOS / Linux
source venv/bin/activate
# Windows
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 3. Run the app

**GUI (recommended):**

```bash
streamlit run app.py
```

This opens automatically in your browser at `http://localhost:8501`. The model
file is ~190 MB, so the **first load takes a few seconds** — after that it's
cached and instant.

**Original CLI**, if you ever want it:

```bash
python cli_app.py
```

---

## 4. Pushing this project to GitHub

⚠️ **Important:** `best_model_rf.pkl` is about **190 MB**, and GitHub blocks any
file over 100 MB on a normal push. You must use **Git LFS** (Large File Storage)
to push it. A `.gitattributes` file is already included in this project to
handle that automatically — just follow the steps below.

### Step 1 — Install Git LFS (one-time, per machine)

```bash
# macOS
brew install git-lfs

# Ubuntu / Debian
sudo apt-get install git-lfs

# Windows
# Download and run the installer from https://git-lfs.com
```

Then enable it for your user account (also one-time):

```bash
git lfs install
```

### Step 2 — Initialize the repo and commit

From inside the `DS_Project` folder:

```bash
git init
git lfs track "*.pkl"          # already set in .gitattributes, but safe to re-run
git add .
git commit -m "Salary Predictor: add GUI app, keep CLI, track model with Git LFS"
```

### Step 3 — Create the GitHub repo

1. Go to [github.com/new](https://github.com/new)
2. Name it (e.g. `salary-predictor`), leave it **empty** (no README/license — you already have files)
3. Click **Create repository**

### Step 4 — Connect and push

GitHub will show you a remote URL — use it here:

```bash
git remote add origin https://github.com/<your-username>/<your-repo>.git
git branch -M main
git push -u origin main
```

The first push will take a little longer because Git LFS uploads the 190 MB
model file separately. You'll see progress like `Uploading LFS objects...`.

### Step 5 — Verify

Open your repo on GitHub and check `best_model_rf.pkl` — it should show a small
text file with an LFS pointer link reading **"Stored with Git LFS"**, and
clicking **Download** should give you the real 190 MB file.

---

## 5. Common issues

| Problem | Fix |
|---|---|
| `git push` rejected: "file exceeds GitHub's file size limit" | Git LFS wasn't tracking the file before your first commit. Run `git lfs track "*.pkl"`, then `git add --renormalize .`, commit again, and push. |
| `ModuleNotFoundError: No module named 'streamlit'` | Run `pip install -r requirements.txt` inside your activated virtual environment. |
| Port `8501` already in use | Run `streamlit run app.py --server.port 8502` |
| App is slow on first prediction | Normal — the 190 MB model is loading and being cached. Every prediction after the first is fast. |
| Cloning the repo elsewhere gives a tiny/broken `.pkl` file | Run `git lfs install` then `git lfs pull` after cloning, so Git LFS actually downloads the real file content. |

---

## 6. About the model

Built from a dataset of job postings with salary, cleaned with an IQR outlier
filter, and compared across Linear Regression vs. Random Forest. Random
Forest won because salary depends on **interactions** between features (e.g.
a senior AI Engineer at a large tech company pays very differently than the
same role at a small retail company) — something a single linear model can't
capture. Full analysis is in `project.ipynb`.
