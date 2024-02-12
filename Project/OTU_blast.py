import os
import sys
import re


#get file names in directory
directory_path = "/home/nyuad/Capstone/Scikit/Selected OTUs"
output_file = "/home/nyuad/Capstone/Scikit/Taxonomy/"

file_list = os.listdir(directory_path)

# Regex pattern to match file names and extract method, number of features, and model type
file_name_pattern = re.compile(r'(.+?) \| (\d+) Features Selected OTUs\.txt')


# Function to read taxonomy data and create a mapping
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

# Function to map OTUs to their taxonomies based on the provided OTU list file
def map_otus_to_taxonomies(otu_list_file_path, taxonomy_mapping):
    with open(otu_list_file_path, 'r') as file:
        otu_list_data = file.readlines()

    otu_ids = [line.strip() for line in otu_list_data if line.strip().isdigit()]
    taxonomy_list = [taxonomy_mapping.get(otu_id, 'Taxonomy not found') for otu_id in otu_ids]
    return taxonomy_list

# Function to save the taxonomy list to a text file
def save_taxonomy_list(taxonomy_list, file_name):
    with open(file_name, 'w') as file:
        for taxonomy in taxonomy_list:
            file.write(f"{taxonomy}\n")
    return file_name

# Main script
if __name__ == "__main__":
    # Check if the necessary arguments are provided
    if len(sys.argv) != 2:
        print("Usage: python script.py <taxonomy_file_path>")
        sys.exit(1)

    taxonomy_file_path = sys.argv[1]
    taxonomy_mapping = create_taxonomy_mapping(taxonomy_file_path)

    for file_name in file_list:
        match = file_name_pattern.match(file_name)
        if match:
            method, num_features = match.groups()
            selected_otus_file_path = os.path.join(directory_path, file_name)
            
            # Process the selected OTUs file and map the OTUs to taxonomies
            taxonomy_list = map_otus_to_taxonomies(selected_otus_file_path, taxonomy_mapping)
            output_file_name = f"{method} | {num_features} Features Taxonomy.txt"
            output_file_path = os.path.join(output_file, output_file_name)
            save_taxonomy_list(taxonomy_list, output_file_path)

            print(f"Taxonomy list for '{method} | {num_features} Features' saved to {output_file_path}")