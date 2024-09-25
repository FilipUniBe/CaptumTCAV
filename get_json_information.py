import pandas as pd
import re
from collections import Counter

# Load the Excel file (replace 'your_file.xlsx' with the actual file name)
# Make sure you specify the correct sheet name or index if it's not the first sheet
file_path = './data/overview-jsons.xlsx'  # Path to your uploaded file

# Load the Excel file into a pandas dataframe
df = pd.read_excel(file_path, header=3)

# Regex to extract the patient and study numbers
patient_study_regex = re.compile(r'patient(\d+)-study(\d+)-')

# Initialize dictionaries for each range for the total across all columns
overall_range_00001_64540 = []
overall_range_64541_64740 = []
overall_range_64741_65240 = []

# List to store entries that don't match the regex
non_matching_entries = []

# Function to categorize numbers into ranges
def categorize_patient_number(patient_number):
    if 1 <= patient_number <= 64540:
        return 'range_00001_64540'
    elif 64541 <= patient_number <= 64740:
        return 'range_64541_64740'
    elif 64741 <= patient_number <= 65240:  # Circular logic (adjust if necessary)
        return 'range_64741_65240'
    return None


# Iterate through all columns in the dataframe
for column in df.columns:
    # Initialize lists for this column's ranges
    range_00001_64540 = []
    range_64541_64740 = []
    range_64741_65240 = []

    # Iterate over each entry in the current column
    for entry in df[column]:
        if isinstance(entry, str):  # Ensure that the entry is a string
            match = patient_study_regex.search(entry)
            if match:
                patient_number = int(match.group(1))
                study_number = int(match.group(2))
                patient_study_str = f'patient{patient_number:05d}-study{study_number}'  # Zero-pad both numbers

                # Categorize the patient number into the correct range
                category = categorize_patient_number(patient_number)
                if category == 'range_00001_64540':
                    range_00001_64540.append(patient_study_str)
                    overall_range_00001_64540.append(patient_study_str)
                elif category == 'range_64541_64740':
                    range_64541_64740.append(patient_study_str)
                    overall_range_64541_64740.append(patient_study_str)
                elif category == 'range_64741_65240':
                    range_64741_65240.append(patient_study_str)
                    overall_range_64741_65240.append(patient_study_str)
            else:
                # Add entry to the list of non-matching entries
                non_matching_entries.append(entry)

    # Output the lists for this column
    print(f"\nPatient numbers for column '{column}':")
    print(f"Range 00001-64540: {sorted(set(range_00001_64540))}")
    print(f"Range 64541-64740: {sorted(set(range_64541_64740))}")
    print(f"Range 64741-64240: {sorted(set(range_64741_65240))}")

# Output the overall totals
print("\nOverall patient numbers across all columns:")
print(f"Range 00001-64540: {sorted(set(overall_range_00001_64540))}")
print(f"Range 64541-64740: {sorted(set(overall_range_64541_64740))}")
print(f"Range 64741-64240: {sorted(set(overall_range_64741_65240))}")

# Output the non-matching entries
print("\nEntries that do not match the regex:")
print(set(non_matching_entries))

# Write the overall valid and test lists to separate text files
with open("./data/train_patient_study.txt", "w") as train_file:
    train_file.write("\n".join(sorted(set(overall_range_00001_64540))))

# Write the overall valid and test lists to separate text files
with open("./data/valid_patient_study.txt", "w") as valid_file:
    valid_file.write("\n".join(sorted(set(overall_range_64541_64740))))

with open("./data/test_patient_study.txt", "w") as test_file:
    test_file.write("\n".join(sorted(set(overall_range_64741_65240))))

