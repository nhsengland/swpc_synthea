import pandas as pd
import os
import argparse
from jinja2 import Environment, FileSystemLoader
from datetime import datetime
from tqdm import tqdm

# CONFIG
DATA_DIR = "output/csv"
OUTPUT_DIR = "output/html_reports"
TEMPLATE_FILE = "contrib/patient-summary/template.html"

def load_and_group_data():
    """
    Loads all CSVs once and groups them by Patient ID for O(1) lookup.
    """
    print(" Loading and grouping data... (This happens only once)")
    
    # 1. Load Patients
    try:
        df_patients = pd.read_csv(f"{DATA_DIR}/patients.csv")
        # Handle cases where Synthea produces dead patients (optional filter)
        # df_patients = df_patients[df_patients['DEATHDATE'].isna()] 
    except FileNotFoundError:
        print(f" Error: Could not find {DATA_DIR}/patients.csv")
        return None, None

    # 2. Define Related Tables
    related_files = {
        "conditions": "conditions.csv",
        "medications": "medications.csv",
        "allergies": "allergies.csv"
    }

    grouped_data = {}

    for name, filename in related_files.items():
        try:
            df = pd.read_csv(f"{DATA_DIR}/{filename}")
            # Logic: Group by PATIENT and convert to a Dictionary of Lists
            # Result: {'uuid-123': [{'DESCRIPTION': 'Diabetes', ...}, {...}], 'uuid-456': ...}
            grouped_data[name] = df.groupby('PATIENT').apply(lambda x: x.to_dict('records')).to_dict()
            print(f"   Loaded {len(df)} {name}")
        except FileNotFoundError:
            print(f"   Warning: {filename} not found.")
            grouped_data[name] = {}
        except Exception as e:
            # Handle cases where a file might be empty or columns missing
            print(f"   Warning: Issue loading {filename}: {e}")
            grouped_data[name] = {}

    return df_patients, grouped_data

def generate_all_reports():
    # 1. Setup
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    df_patients, clinical_data = load_and_group_data()
    if df_patients is None:
        return

    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template(TEMPLATE_FILE)
    
    generated_files = []

    # 2. The Loop (Iterate through all patients)
    print(f"Generating reports for {len(df_patients)} patients...")
    
    # Use tqdm if available, else standard iterator
    iterator = tqdm(df_patients.iterrows(), total=len(df_patients)) if 'tqdm' in globals() else df_patients.iterrows()

    for _, patient in iterator:
        p_dict = patient.to_dict()
        p_id = p_dict['Id']
        
        # Calculate Age
        try:
            birth_date = datetime.strptime(p_dict['BIRTHDATE'], "%Y-%m-%d")
            age = datetime.now().year - birth_date.year
        except:
            age = "N/A"

        # 3. Render
        html_out = template.render(
            p=p_dict,
            age=age,
            # Safe Lookup: Get list for this ID, or empty list if key missing
            conditions=clinical_data['conditions'].get(p_id, []),
            medications=clinical_data['medications'].get(p_id, []),
            allergies=clinical_data['allergies'].get(p_id, []),
            gen_date=datetime.now().strftime("%d-%m-%Y")
        )

        # 4. Save
        filename = f"summary_{p_dict['FIRST']}_{p_dict['LAST']}_{p_id[-4:]}.html"
        with open(f"{OUTPUT_DIR}/{filename}", "w", encoding="utf-8") as f:
            f.write(html_out)
            
        generated_files.append({"name": f"{p_dict['FIRST']} {p_dict['LAST']}", "file": filename, "id": p_id})

    # 5. Generate Master Index
    generate_index_page(generated_files)
    print(f"\n Done! Reports saved in {OUTPUT_DIR}/")

def generate_index_page(files):
    """
    Creates a simple 'Home Page' to browse all patients.
    """
    html = """
    <html>
    <head>
        <title>Patient Directory</title>
        <style>
            body { font-family: sans-serif; padding: 20px; background: #f0f4f5; }
            .card { background: white; padding: 15px; margin-bottom: 10px; border-radius: 5px; box-shadow: 0 2px 3px #ddd; }
            a { text-decoration: none; color: #005EB8; font-weight: bold; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <h1>Patient Directory</h1>
        <p>Total Patients: """ + str(len(files)) + """</p>
    """
    
    for f in files:
        html += f'<div class="card"><a href="{f["file"]}">{f["name"]}</a> <small>(ID: ...{f["id"][-6:]})</small></div>'
    
    html += "</body></html>"
    
    with open(f"{OUTPUT_DIR}/index.html", "w", encoding="utf-8") as f:
        f.write(html)

if __name__ == "__main__":
    generate_all_reports()