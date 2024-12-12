from utils.patient_plots import plot_bp, create_sankey_diagram
from glob import glob
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt 

# Point this to the csv subfolder of the output directory created by Synthea
output_path = "../Sample_data/csv_hypertension_1000"

conds = pd.read_csv(f"{output_path}/conditions.csv")
obs = pd.read_csv(f"{output_path}/observations.csv")
meds = pd.read_csv(f"{output_path}/medications.csv")
pats = pd.read_csv(f"{output_path}/patients.csv")
encounters = pd.read_csv(f"{output_path}/encounters.csv")

st.header("Summary Statistics on the Data")

csvs = glob(f"{output_path}/*csv")



col1, col2 = st.columns(2)

with col1:     
    ht_conds = conds[conds["DESCRIPTION"] == "Essential hypertension (disorder)"]
    patients_count = len(list(pats["Id"]))
    ht_patients = list(set(conds[conds["DESCRIPTION"] == "Essential hypertension (disorder)"]["PATIENT"]))
    num_ht = len(ht_patients)

    fig, ax = plt.subplots(1,1, figsize=(6,3))
    ax.bar(['Hypertension', 'No Hypertension'], [num_ht, patients_count-num_ht], color="lightcoral")

    title = f"Number of Patients with and without hypertension"

    ax.set_title(title, fontdict={'fontsize':10})
    plt.legend()
    st.pyplot(fig, use_container_width=False)  

    genders = pats.GENDER.value_counts()

    fig, ax = plt.subplots(figsize=(8, 4))
    genders.plot(kind='bar', ax=ax, color='mediumpurple')
    plt.xticks(rotation=0)

    ax.set_title("Distribution of patients by gender.")

    st.pyplot(fig)  

    

with col2:
    # print(pats.keys())
    ethnicities = pats.RACE.value_counts()

    fig, ax = plt.subplots(figsize=(8, 4))
    ethnicities.plot(kind='bar', ax=ax, color='skyblue')
    plt.xticks(rotation=0)

    ax.set_title("Distribution of patients by ethnicity.")

    st.pyplot(fig)  

    patient_number = st.number_input(
        f"Pick a patient to see their blood pressure over time, there are \
            {num_ht} patients to choose from.",
        min_value=0,
        max_value=num_ht
        )
    
    plot_bp(ht_patients[patient_number], obs, conds, meds)

filtered_dfs = []


for file in csvs:
    df = pd.read_csv(file)
    
    if 'PATIENT' in df.columns:
        filtered_df = df[df['PATIENT'] == ht_patients[patient_number]]
        filtered_df['source_table'] = file.split('/')[-1].replace('.csv', '')
        filtered_dfs.append(filtered_df)

final_df = pd.concat(filtered_dfs, ignore_index=True)
# final_df = final_df[(final_df["START"].notnull())]

final_df = final_df[~(final_df["DESCRIPTION"].isin(
    ["Well child visit (procedure)", "General examination of patient (procedure)", 
    'QALY', 'QOLS', 'DALY', 'Systolic Blood Pressure']))]


fig, ax = plt.subplots(figsize=(8, 4))

# Drop rows with missing START or DESCRIPTION
final_df['DATE'] = pd.to_datetime(final_df['DATE'], errors='coerce')

final_df['START'] = final_df['START'].fillna(final_df['DATE'])

final_df['START'] = pd.to_datetime(final_df['START'], errors='coerce')


timeline_df = final_df.dropna(subset=['START', 'DESCRIPTION'])

# Sort by START date
timeline_df = timeline_df.sort_values(by='START')

def plot_timeline(df):
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot points on timeline
    ax.plot(df['START'], [1] * len(df),linestyle='-',marker='o', color="C0")  # Points on the same y-axis level

    # Annotate each point with the DESCRIPTION label
    for i, row in df.iterrows():
        ax.text(row['START'], 1.01, row['DESCRIPTION'], rotation=70, ha='left', fontsize=8)
        ax.plot([row['START'], row['START']], [1, 1.01], color='gray', 
                linestyle='--', linewidth=0.5)

    # Format plot
    ax.set_yticks([])  # Hide y-axis
    ax.set_xlabel("Date")
    ax.set_title("Timeline of Events for Patient")
    ax.set_ylim(0.9,1.1)
    plt.xticks(rotation=45)

    return fig

timeline_plot = plot_timeline(timeline_df)
st.pyplot(timeline_plot)

meds_ht = meds[meds['REASONCODE']=='Essential hypertension (disorder)']['PATIENT'].nunique()

total_no_hypertension_patients = patients_count - num_ht
# medicated_dead = 20, 
# medicated_alive = 730, unmedicated_dead = 10,
# unmedicated_alive = 40, no_hypertension_alive = 190,
# no_hypertension_dead = 10
# ht_patients = list(set(conds[conds["DESCRIPTION"] == "Essential hypertension (disorder)"]["PATIENT"]))

st.write(ht_patients)

st.plotly_chart(
    create_sankey_diagram(total_hypertension_patients=num_ht, 
                        total_no_hypertension_patients= total_no_hypertension_patients,
                        total_medicated=meds_ht, 
                        total_unmedicated=total_no_hypertension_patients - meds_ht)
                )

current_date = pd.Timestamp.now()

ht_conds['START'] = pd.to_datetime(ht_conds['START'])
ht_conds['STOP'] = pd.to_datetime(ht_conds['STOP'])

pats_death = pats[['Id', 'DEATHDATE']]
ht_conds_with_death = pd.merge(ht_conds, pats, left_on='PATIENT', right_on='Id', how='left')

ht_conds_with_death['STOP'] = ht_conds_with_death['STOP'].fillna(ht_conds_with_death['DEATHDATE'])
ht_conds_with_death['STOP'] = ht_conds_with_death['STOP'].fillna(current_date)


# Create a DataFrame with all unique dates
all_dates = pd.date_range(start=ht_conds_with_death['START'].min(), end=ht_conds_with_death['STOP'].max())
date_df = pd.DataFrame(all_dates, columns=['DATE'])

# Calculate cumulative count of patients with hypertension for each date
date_df['CUMULATIVE_COUNT'] = date_df['DATE'].apply(
    lambda x: ((ht_conds_with_death['START'] <= x) & (ht_conds_with_death['STOP'] >= x)).sum()
)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(date_df['DATE'], date_df['CUMULATIVE_COUNT'], marker=None, linestyle='-', linewidth=2)

# Set labels and title
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Count of Patients with Hypertension')
ax.set_title('Cumulative Count of Patients with Hypertension Over Time')

# Rotate x-ticks for better readability
plt.xticks(rotation=45)

st.pyplot(fig)