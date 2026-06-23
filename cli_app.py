# cli_app.py — Original CLI Prediction App
# How it works
# Load pickled model + encoders → prompt user for 9 inputs → encode → predict → display result.
# Run with: python cli_app.py
#
# Kept for reference / terminal use. The GUI version lives in app.py.

import joblib, pandas as pd, sys, os

# ── Load Artefacts ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
rf_model = joblib.load(os.path.join(BASE_DIR, "best_model_rf.pkl"))
label_encoders = joblib.load(os.path.join(BASE_DIR, "label_encoders.pkl"))

FEATURES = ["job_title","experience_years","education_level",
            "skills_count","industry","company_size",
            "location","remote_work","certifications"]

def predict_salary(**kwargs):
    row = dict(kwargs)
    cat = ["job_title","education_level","industry",
           "company_size","location","remote_work"]
    for col in cat:
        le = label_encoders[col]
        row[col] = le.transform([row[col]])[0]
    X = pd.DataFrame([row])[FEATURES]
    return round(rf_model.predict(X)[0], 2)

def run_cli():
    opts = {k: list(v.classes_) for k,v in label_encoders.items()}

    def ask(prompt, valid=None, cast=str):
        while True:
            val = input(f"  {prompt}: ").strip()
            if cast in (int,float):
                try: return cast(val)
                except: print("  ⚠️  Invalid input. Please enter a number.")
            elif valid and val not in valid:
                print(f"  ⚠️  Invalid choice. Please choose from: {valid}")
            else: return val

    print("\n" + "="*40)
    print("   SALARY PREDICTION CONTROL PANEL")
    print("="*40)
    
    jt  = ask(f"Job Title {opts['job_title']}",        opts["job_title"])
    exp = ask("Experience years (0-20)",              cast=int)
    edu = ask(f"Education {opts['education_level']}",  opts["education_level"])
    sk  = ask("Skills count (1-19)",                  cast=int)
    ind = ask(f"Industry {opts['industry']}",          opts["industry"])
    cs  = ask(f"Company size {opts['company_size']}",  opts["company_size"])
    loc = ask(f"Location {opts['location']}",          opts["location"])
    rw  = ask(f"Remote work {opts['remote_work']}",    opts["remote_work"])
    cer = ask("Certifications (0-5)",                 cast=int)

    salary = predict_salary(
        job_title=jt, experience_years=exp, education_level=edu,
        skills_count=sk, industry=ind, company_size=cs,
        location=loc, remote_work=rw, certifications=cer)
    
    print("\n" + "-"*40)
    print(f"  RESULT: Predicted Salary is ${salary:,.2f}")
    print("-"*40 + "\n")

if __name__ == "__main__":
    run_cli()
