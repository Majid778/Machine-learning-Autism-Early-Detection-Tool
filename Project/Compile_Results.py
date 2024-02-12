import re
import os
import pandas as pd

# Directory where the files are located (assuming all files are in the same directory)
directory_path = '/home/nyuad/Capstone/Scikit/Evaluation Results/'

# List of file names (replace with the actual file names)
#get addresses of all files in the directory

file_list = os.listdir(directory_path)
for file_name in file_list:
    print(file_name)

# Regex pattern to match file names and extract method, number of features, and model type
file_name_pattern = re.compile(r'(.+?) \| (\d+) Features \| (.+?) Evaluation Results\.txt')

# Regex pattern to extract metrics from the file content
metrics_pattern = re.compile(
    r'Correctly Classified Instances:\s+(?P<Correctly_Classified_Instances>\d+)|'
    r'Incorrectly Classified Instances:\s+(?P<Incorrectly_Classified_Instances>\d+)|'
    r'Accuracy:\s+(?P<Accuracy>\d+\.\d+)|'
    r'Kappa statistic:\s+(?P<Kappa_statistic>\d+\.\d+)|'
    r'Mean absolute error:\s+(?P<Mean_absolute_error>\d+\.\d+)|'
    r'Root mean squared error:\s+(?P<Root_mean_squared_error>\d+\.\d+)|'
    r'Relative absolute error:\s+(?P<Relative_absolute_error>\d+\.\d+)|'
    r'Root relative squared error:\s+(?P<Root_relative_squared_error>\d+\.\d+)|'
    r'Precision:\s+(?P<Precision>\d+\.\d+)|'
    r'Recall:\s+(?P<Recall>\d+\.\d+)|'
    r'F1 Score:\s+(?P<F1_Score>\d+\.\d+)|'
    r'Total Number of Instances:\s+(?P<Total_Number_of_Instances>\d+)|'
    r"Confusion Matrix: \{'True Negative': (?P<True_Negative>\d+), 'False Positive': (?P<False_Positive>\d+), 'False Negative': (?P<False_Negative>\d+), 'True Positive': (?P<True_Positive>\d+)\}|"
    r'AUC:\s+(?P<AUC>\d+\.\d+)'
)


# Function to extract metrics from file content
def extract_metrics(content, pattern):
    matches = pattern.finditer(content)
    metrics = {}
    for match in matches:
        metrics.update({k: v for k, v in match.groupdict().items() if v is not None})
    return metrics

# List to hold the aggregated results
aggregated_results = []

# Process each file
for file_name in file_list:
    # Match the file name to extract the method, number of features, and model type
    file_name_match = file_name_pattern.match(file_name)
    print(file_name_match)
    if file_name_match:
        method, num_features, model = file_name_match.groups()
        file_path = os.path.join(directory_path, file_name)
        
        # Read the content of the file
        with open(file_path, 'r') as file:
            content = file.read()
        
        # Extract metrics from the content
        metrics = extract_metrics(content, metrics_pattern)
        
        # Add the extracted data to the results dictionary
        result = {
            'Method': method,
            'Number of Features': num_features,
            'Model': model
        }
        result.update(metrics)
        
        # Add the dictionary to our results list
        aggregated_results.append(result)

# Convert the aggregated results to a DataFrame
results_df = pd.DataFrame(aggregated_results)

#sort by method, then by model, then by number of features
results_df = results_df.sort_values(['Method', 'Model', 'Number of Features'])
# Save the DataFrame to a CSV file
csv_file_path = directory_path+'/evaluation_results.csv'
results_df.to_csv(csv_file_path, index=False)
