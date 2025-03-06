import os
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

# Parameters
naca_codes = [f"{i:04}" for i in np.arange(1,30)]  # NACA airfoils 0000 to 9999
mach = 10 / 340 # Mach number
reynolds = 67680  # Reynolds number
alpha_range = (0, 15)  # AoA range
alpha_step = 1 # Step between two alphas

# Path to XFOIL
xfoil_path = "xfoil" # If reachable from path

# Output folders
polars_folder = "./polars/"
output_folder = './output/'
os.makedirs(polars_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# Score weights
endurance_score_weights = {'CL': 0.4, 'CD': -0.5, 'CL/CD': 1, 'CM': -0.3}
payload_score_weights = {'CL': 1, 'CD': -0.2, 'CL/CD': 0.6, 'CM': -0.1}

# Number of best airfoils' polars to plot
plots_count = 10

# Template for XFOIL input
with open('template_script.txt', 'r') as file:
    template_script = file.read()

# Function to run XFOIL for a single airfoil
def run_xfoil(naca_code):
    script_file = polars_folder + f"xfoil_script_{naca_code}.txt"
    polar_file = polars_folder + f"naca{naca_code}_polar.dat"
    script = template_script.format(code=naca_code,
                                    reynolds=reynolds,
                                    mach=mach,
                                    a1=alpha_range[0],
                                    a2=alpha_range[1],
                                    step=alpha_step,
                                    output=polar_file)
    
    with open(script_file, 'w') as file:
        file.write(script)
    
    subprocess.run([xfoil_path], input=script.encode(), capture_output=True)
    
    if os.path.exists(polar_file):
        data = parse_polar_file(polar_file)
        os.remove(polar_file)
        os.remove(script_file)
        return (naca_code, data)
    else:
        print("No data found.")
        return (naca_code, None)

# Function to parse polar data
def parse_polar_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    data = []
    parsing = False
    for line in lines:
        if 'alpha' in line.lower():
            parsing = True
            continue
        if parsing:
            try:
                values = list(map(float, line.split()))
                data.append(values)
            except ValueError:
                continue
    
    df = pd.DataFrame(data, columns=['Alpha', 'CL', 'CD', 'CDp', 'CM', 'Top_Xtr', 'Bot_Xtr'])
    df['CL/CD'] = df['CL'] / df['CD']

    # Scoring function
    def compute_score(row, weights):
        score = sum(row[col] * weight for col, weight in weights.items())
        return score

    # Add score and average score columns
    df['Endurance_Score'] = df.apply(lambda row: compute_score(row, endurance_score_weights), axis=1)
    df['Payload_Score'] = df.apply(lambda row: compute_score(row, payload_score_weights), axis=1)
    df['Average_Endurance_Score'] = df['Endurance_Score'].mean()
    df['Average_Payload_Score'] = df['Payload_Score'].mean()

    return df

# Run simulations in parallel
def main():
    with Pool() as pool:
        results = pool.map(run_xfoil, naca_codes)
    
    all_data = {code: data for code, data in results if data is not None}
    
    combined_data = pd.concat([df.assign(Airfoil=code) for code, df in all_data.items()], ignore_index=True)
    combined_data.to_csv(output_folder + 'all_airfoils_data.csv', index=False)
    
    top_airfoils = get_top_airfoils(combined_data)
    plot_polar_curves(combined_data, top_airfoils)
    plot_score_scatter(combined_data)

# Get the most interesting airfoils
def get_top_airfoils(data):
    avg_scores = data.groupby('Airfoil')[['Average_Endurance_Score', 'Average_Payload_Score']].mean().reset_index()
    top_endurance = avg_scores.nlargest(plots_count // 2, 'Average_Endurance_Score')
    top_payload = avg_scores.nlargest(plots_count // 2, 'Average_Payload_Score')
    return pd.concat([top_endurance, top_payload])['Airfoil'].unique()

# Plotting function
def plot_polar_curves(data, top_airfoils):
    for airfoil, df in data.groupby('Airfoil'):
        if airfoil not in top_airfoils:
            continue

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        
        axs[0, 0].plot(df['Alpha'], df['CL'], color='b')
        axs[0, 0].set_title('CL vs Alpha')
        axs[0, 0].set_xlabel('Angle of Attack (°)')
        axs[0, 0].set_ylabel('CL')
        axs[0, 0].grid(True)

        axs[0, 1].plot(df['Alpha'], df['CD'], color='r')
        axs[0, 1].set_title('CD vs Alpha')
        axs[0, 1].set_xlabel('Angle of Attack (°)')
        axs[0, 1].set_ylabel('CD')
        axs[0, 1].grid(True)

        axs[1, 0].plot(df['Alpha'], df['CL/CD'], color='g')
        axs[1, 0].set_title('CL/CD vs Alpha')
        axs[1, 0].set_xlabel('Angle of Attack (°)')
        axs[1, 0].set_ylabel('CL/CD')
        axs[1, 0].grid(True)

        axs[1, 1].plot(df['Alpha'], df['CM'], color='purple')
        axs[1, 1].set_title('CM vs Alpha')
        axs[1, 1].set_xlabel('Angle of Attack (°)')
        axs[1, 1].set_ylabel('CM')
        axs[1, 1].grid(True)

        fig.suptitle(f'NACA {airfoil} Polars')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_folder + f'naca{airfoil}_polars.png')
        plt.close()

# Scatter plot for comparing airfoils by scores
def plot_score_scatter(data):
    avg_scores = data.groupby('Airfoil')[['Average_Endurance_Score', 'Average_Payload_Score']].mean().reset_index()
    
    top_endurance = avg_scores.nlargest(plots_count // 2, 'Average_Endurance_Score')
    top_payload = avg_scores.nlargest(plots_count // 2, 'Average_Payload_Score')
    highlighted = pd.concat([top_endurance, top_payload]).drop_duplicates()

    plt.figure(figsize=(10, 6))
    plt.scatter(avg_scores['Average_Endurance_Score'], avg_scores['Average_Payload_Score'], color='lightgray')
    plt.scatter(highlighted['Average_Endurance_Score'], highlighted['Average_Payload_Score'], color='red')
    
    for _, row in highlighted.iterrows():
        plt.annotate(row['Airfoil'], (row['Average_Endurance_Score'], row['Average_Payload_Score']))
    
    plt.title('Top Payload vs Endurance Scores for NACA Airfoils')
    plt.xlabel('Average Endurance Score')
    plt.ylabel('Average Payload Score')
    plt.grid(True)
    plt.savefig(output_folder + 'top_score_comparison_scatter.png')
    plt.close()

if __name__ == "__main__":
    main()
