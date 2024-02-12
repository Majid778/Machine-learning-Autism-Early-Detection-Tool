import os
import re
import pandas as pd
from scipy.stats import ttest_ind
import numpy as np

# Define your directory paths and file patterns
directory_path = "/home/nyuad/Capstone/Scikit/Selected OTUs"
output_directory = "/home/nyuad/Capstone/Scikit/Taxonomy"
taxonomy_file_path = "/home/nyuad/Capstone/Scikit/Taxonomy/97_otu_taxonomy.txt"
file_pattern = re.compile(r'(.+?) \| (\d+) Features Selected OTUs\.txt')
otu_data_path = "/home/nyuad/Capstone/Scikit/master_otu.tsv"  # Path to the OTU abundance data

# Function to create taxonomy mapping
def create_taxonomy_mapping(taxonomy_file_path):
    with open(taxonomy_file_path, 'r') as file:
        taxonomy_data = file.readlines()

    taxonomy_mapping = {}
    for line in taxonomy_data:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            otu_id, taxonomy = parts
            taxonomy_mapping[otu_id] = taxonomy
    return taxonomy_mapping

# Function to perform differential analysis and calculate log2 fold change
def perform_differential_analysis_and_log2_fold_change(otu_list, otu_data):
    results = {}
    for otu in otu_list:
        control_group = otu_data[otu_data['Class'] == 0][otu]
        treatment_group = otu_data[otu_data['Class'] == 1][otu]
        
        # Calculate t-statistic and p-value
        t_stat, p_value = ttest_ind(control_group, treatment_group, equal_var=False, nan_policy='omit')
        
        # Calculate means for the control and treatment groups
        control_mean = np.mean(control_group)
        treatment_mean = np.mean(treatment_group)
        
        # Avoid division by zero by adding a small number (epsilon) to control_mean
        epsilon = 1e-10
        fold_change = (treatment_mean + epsilon) / (control_mean + epsilon)
        log2_fold_change = np.log2(fold_change)
        
        results[otu] = {
            't_stat': t_stat, 
            'p_value': p_value, 
            'log2_fold_change': log2_fold_change
        }
    return pd.DataFrame.from_dict(results, orient='index')

# Main script
taxonomy_mapping = create_taxonomy_mapping(taxonomy_file_path)

# Read tsv file 
otu_data = pd.read_csv(otu_data_path, sep="\t")

file_list = os.listdir(directory_path)

for file_name in file_list:
    match = file_pattern.match(file_name)
    if match:
        selected_otus_file_path = os.path.join(directory_path, file_name)
        with open(selected_otus_file_path, 'r') as file:
            selected_otus = [line.strip() for line in file.readlines()]

        # Perform differential analysis and calculate log2 fold change
        da_results = perform_differential_analysis_and_log2_fold_change(selected_otus, otu_data)

        # Add taxonomy information
        da_results['Taxonomy'] = da_results.index.map(lambda otu: taxonomy_mapping.get(str(otu), 'Taxonomy not found'))

        # Make taxonomy column the second column
        cols = da_results.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        da_results = da_results[cols]

        # Filter for significant results
        significant_results = da_results[da_results['p_value'] <= 0.05]

        # Strip file name
        file_name = file_name.replace(".txt", ".csv")

        # Save the results
        output_file_path = os.path.join(output_directory, f"Differential Analysis | {file_name}")
        significant_results.to_csv(output_file_path, index=True)

        print(f"Taxonomy integrated results saved to {output_file_path}")
