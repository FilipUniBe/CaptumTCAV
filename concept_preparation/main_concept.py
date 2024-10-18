'''call this, do all'''
import json
import os
import pickle
import shutil
import itertools
from collections import Counter, defaultdict

import pandas as pd

from concept_preparation.get_negative_concepts import get_negative_concepts
from concept_preparation.get_concepts import extract_components, extract_data, get_concepts

import logging
from logging_config import setup_logging

# importing sys
from model.init import load_config
import sys
sys.path.insert(0, '../tcav')


def filter_grouped_files(grouped_files, instructions):
    filtered_keys = {}

    # Iterate through the keys and values in the grouped_files dictionary
    for key, value in grouped_files.items():
        # Initialize a flag to check if all instructions match
        match = True

        # Check if the key exists in the instructions
        if key in instructions:
            # Check if the instruction is "*" (wildcard)
            if isinstance(instructions[key], list):
                # If the instruction is a list, check if any value in the list matches
                if not any(item in instructions[key] for item in value):
                    match = False
            else:
                # Check if the value matches the instruction
                if instructions[key] not in value:
                    match = False
            if instructions[key] == "*":
                instructions[key] = list(value.keys())
                match = True
        else:
            match = False

        # If all instructions match, add the group to the filtered keys
        if match:
            if isinstance(instructions[key], list):
                # If instruction is a list, add all matching values to filtered_keys
                filtered_keys[key] = [value[single_key] for single_key in instructions[key]]
            else:
                # If instruction is a single value, add the matching value to filtered_keys
                filtered_keys[key] = value[instructions[key]]

    return filtered_keys
def group_concepts(folder_path):
    '''Keys according to README.md in data/'''
    # Initialize a dictionary to store the grouped files
    grouped_files = {}

    # Iterate through the files in the folder
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            # Split the filename into parts based on underscore
            parts = filename.split("_")
            # Extract key-value pairs from filename
            file_info = {}
            for part in parts:
                key_value = part.split("-")
                if len(key_value) == 2:
                    file_info[key_value[0]] = key_value[1]
            # Group the files based on keys
            for key, value in file_info.items():
                if key not in grouped_files:
                    grouped_files[key] = {}
                if value not in grouped_files[key]:
                    grouped_files[key][value] = []
                grouped_files[key][value].append(filename)

    return grouped_files

def generate_combinations(instructions):
    """Generate all possible combinations of values for each key in instructions."""
    keys = instructions.keys()
    value_combinations = [instructions[key] if isinstance(instructions[key], list) else [instructions[key]] for key in
                          keys]
    return list(itertools.product(*value_combinations))


def flatten_lists(dictionary):
    flattened_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, list):
            # If the value is a list, flatten it
            flattened_list = []
            for item in value:
                if isinstance(item, list):
                    # If the item in the list is also a list, extend the flattened list
                    flattened_list.extend(item)
                else:
                    # Otherwise, append the item to the flattened list
                    flattened_list.append(item)
            flattened_dict[key] = flattened_list
        elif isinstance(value, dict):
            # If the value is a dictionary, recursively flatten it
            flattened_dict[key] = flatten_lists(value)
        else:
            # Otherwise, keep the value as it is
            flattened_dict[key] = value
    return flattened_dict


def create_folder_for_filtered_files(filtered_filenames, folder_path):
    """Create a folder and move the filtered files into that folder."""
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)

    # Copy each filtered file into the folder
    for filename in filtered_filenames:
        source =filename
        destination = os.path.join(folder_path, os.path.basename(filename))
        shutil.copy(source, destination)

def common_elements(dictionary):
    # Get the values (lists) from the dictionary
    lists = list(dictionary.values())

    # Take the intersection of all lists
    common_elements = set.intersection(*map(set, lists))

    # Convert the set to a list
    return list(common_elements)

def get_group_identifier(file_path):
    # Extract the filename from the path
    filename = os.path.basename(file_path)
    # Remove the patient/study/view part
    parts = filename.split("_")

    # Remove elements containing "nr-00X"
    parts = [part for part in parts if not part.startswith("nr-")]
    parts = [part for part in parts if not part.startswith("filename-")]

    group_identifier = "_".join(parts[:-1])

    return group_identifier

# Function to collect files to copy based on group identifier
def collect_files_to_copy(base_dir):
    files_to_copy = {}

    for root, _, files in os.walk(base_dir):
        for file in files:
            if "type-neg" in file:
                continue  # Skip files with "type-neg" in the filename
            if file.endswith(".jpg"):  # Assuming only processing .jpg files
                file_path = os.path.join(root, file)
                group_identifier = get_group_identifier(file_path)
                if group_identifier in files_to_copy:
                    files_to_copy[group_identifier].append(file_path)
                else:
                    files_to_copy[group_identifier] = [file_path]

    return files_to_copy

# Function to group files by their identifier and copy them to destination folders
def group_files_and_copy(files_to_copy, dest_dir,force_overwrite):
    file_counts = {}

    for group_identifier, file_paths in files_to_copy.items():

        # Modify the group_identifier to ignore the json tag unless it's json-1
        if "json-1" not in group_identifier:
            group_identifier = "_".join([part for part in group_identifier.split('_') if not part.startswith('json-')])

        # Catch bullshit I don't know where it comes from
        if "healthy_train" in group_identifier:
            continue
        if "random_train" in group_identifier:
            continue

        group_path = os.path.join(dest_dir, group_identifier)

        # Create the destination group folder if it doesn't exist
        if not os.path.exists(group_path):
            os.makedirs(group_path)

        # Copy files to the group folder
        for file_path in file_paths:
            filename = os.path.basename(file_path)
            try:
                dest_file_path = os.path.join(group_path, filename)
                if os.path.exists(dest_file_path):
                    if not force_overwrite:
                        logging.info(f"Skipping {dest_file_path} (already exists)")
                        continue
                shutil.copy(file_path, group_path)
                # Update the file count for the group
                if group_identifier in file_counts:
                    file_counts[group_identifier] += 1
                else:
                    file_counts[group_identifier] = 1
            except shutil.SameFileError:
                print(f"Skipping copy: {file_path} already exists in {group_path}")
        print(f"copied over: {group_identifier}")

    # Generate the report
    print("File grouping report:")
    for group, count in file_counts.items():
        print(f"{group}: {count} files")

def prepare_concept_folders(config_folder,force_overwrite):
    # Define the folder path
    folder_path = config_folder

    # to group all files into all available keys. (multiples possible)
    grouped_files = group_concepts(folder_path)

    # Collect files to copy based on group identifiers
    files_to_copy = collect_files_to_copy(concept_folder)

    # Group files and copy them to destination folders
    group_files_and_copy(files_to_copy, concept_folder,force_overwrite)

    process_negative_files(grouped_files, folder_path)

def add_basepath(filtered_files, base_path):
    """Add a base path to each filename in the list."""
    return [os.path.join(base_path, filename) for filename in filtered_files]
def process_negative_files(grouped_files, base_folder, max_files_per_folder=50):
    # Extract negative files
    negative_files = grouped_files.get("type", {}).get("neg", [])

    # Create folders for negative files
    create_folder_for_neg_filtered_files(negative_files, base_folder, max_files_per_folder)

def create_folder_for_neg_filtered_files(filtered_filenames, folder_path, max_files_per_folder=50):
    """Create a folder and copy the filtered files into that folder, with a limit on the number of files per folder."""
    folder_index = 0
    file_count = 0

    # Create the base folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    # Create subfolders and copy files into them
    for i, filename in enumerate(filtered_filenames):
        if file_count % max_files_per_folder == 0:
            subfolder_path = os.path.join(folder_path, f"negatives_{folder_index}")
            try:
                # Create the folder if it doesn't exist
                os.makedirs(subfolder_path, exist_ok=True)
                print(f"Successfully created the directory: {subfolder_path}")
            except Exception as e:
                print(f"Failed to create the directory: {subfolder_path}")
                print(f"Error: {e}")
            folder_index += 1

        source = filename
        base_path = folder_path
        source_with_base_path = os.path.join(base_path, source)
        destination = os.path.join(subfolder_path, os.path.basename(filename))

        shutil.copy(source_with_base_path, destination)
        file_count += 1

def get_concept_image_count(base_dir):
    # Collect files to copy based on group identifiers
    # Dictionary to hold grouped filenames and their tag variations
    grouped_files = defaultdict(list)

    for filename in os.listdir(base_dir):
        if filename.endswith(".jpg"):  # Ensure you're only processing .jpg files
                parts = filename.split("_")
                group_identifier = "_".join(parts[:-1])
                grouped_files[group_identifier].append(filename)

    # Dictionary to hold tag counts per group_identifier
    tag_counts = defaultdict(lambda: defaultdict(int))

    # Count the tag variations for each group
    for group_identifier, filenames in grouped_files.items():
        for filename in filenames:
            #different approach for na-files
            if "na" in filename:
                continue
            type_key, abbr_key, exp_key, form_key, resize_key, bg_key, nr_key, json_file_number, original_filename = extract_components(
                filename)
            tag_key = (type_key, abbr_key, exp_key, form_key, resize_key, bg_key, nr_key, json_file_number)
            tag_counts[group_identifier][tag_key] += 1

    # Convert the nested defaultdict to a regular dictionary for cleaner output
    tag_counts = {group: dict(counts) for group, counts in tag_counts.items()}

    return tag_counts

def get_processed_data_pickle_count():
    pickle_file = "processed_data.pkl"
    if os.path.exists(pickle_file):
        logging.info(f"Loading data from existing pickle file: {pickle_file}")
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f)
            # Group all entries by entry["json_file"]
        json_file_groups = defaultdict(lambda: defaultdict(int))
        for entry in pickle_data:
            json_file = entry["json_file"]
            filename = entry["filename"]
            json_file_groups[json_file][filename] += 1

        # Convert the nested defaultdict to a regular dictionary for cleaner output
        json_file_groups = {json_file: dict(filenames) for json_file, filenames in json_file_groups.items()}

    else:
        logging.warning(f"Pickle file does not exist: {pickle_file}")

    return json_file_groups

def extract_tags(foldername):
    """
    Extract tags and their values from the filename.
    """
    parts = foldername.split("_")
    tags = {}
    for part in parts:
        tag, value = part.split("-", 1)
        tags[tag] = value
    return tags

def count_tags(base_dir):
    # Dictionary to hold the count of each tag-value pair
    tag_counts = defaultdict(lambda: defaultdict(int))
    folders = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            folders.append(item)
    for folder in folders:
        #skip for now
        if "negatives" in folder:
            continue
        tags = extract_tags(folder)
        for tag, value in tags.items():
            tag_counts[tag][value] += 1
    return tag_counts


def aggregate_data(new_data):
    aggregated_data = defaultdict(lambda: defaultdict(Counter))

    for entry in new_data:
        json_file = entry['json_file']
        filename = entry['filename']
        annotation_form = entry['annotation_form']

        aggregated_data[json_file]['unique_filenames'].add(filename)
        aggregated_data[json_file][filename]['annotation_forms'][annotation_form] += 1

    return aggregated_data
def generate_report(concept_folder, json_files):
    report_lines = []

    print('\n')
    print('#### check untouched json for data ###')
    print('\n')
    # count unprocessed annotation-data
    all_filenames = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            filenames=data["_via_image_id_list"]
            # List of filenames to exclude
            exclude_filenames = ["adutta_swan.jpg", "wikimedia_death_of_socrates.jpg"]
            # Filter out filenames based on the exclusion list
            filtered_filenames = [filename for filename in filenames if filename not in exclude_filenames]
            print(f"Filenames in {json_file}: {len(filtered_filenames)}")
            all_filenames.extend(filtered_filenames)
    # Counting unique filenames
    unique_filenames_count = len(set(all_filenames))
    # Counting non-unique filenames
    filenames_counter = Counter(all_filenames)
    non_unique_filenames_count = sum(count > 1 for count in filenames_counter.values())
    print(f"Total unique filenames: {unique_filenames_count}")
    print(f"Total non-unique filenames: {non_unique_filenames_count}")

    print('\n')
    print('#### check processed json ###')
    print('\n')

    #counte extracted data
    new_data = extract_data(json_files)

    # Initialize an empty dictionary to store grouped data
    grouped_data = {}

    # Iterate through new_data and group dictionaries by json_file number
    for entry in new_data:
        json_file_number = entry["json_file"]
        filename = entry["filename"]
        abbr = entry["region_type"]

        # Skip specific filenames
        if filename=="adutta_swan.jpg" or filename == "wikimedia_death_of_socrates.jpg":
            continue #default images in project that we don't need

        if json_file_number not in grouped_data:
            grouped_data[json_file_number] = {
            "unique_abbr": set(),
            "abbr_count": {}
        }
        if filename not in grouped_data[json_file_number]:
            grouped_data[json_file_number][filename] = {}
        if abbr not in grouped_data[json_file_number][filename]:
            grouped_data[json_file_number][filename][abbr] = []
        if abbr not in grouped_data[json_file_number]["unique_abbr"]:
            grouped_data[json_file_number]["unique_abbr"].add(abbr)

        if abbr not in grouped_data[json_file_number]["abbr_count"]:
            grouped_data[json_file_number]["abbr_count"][abbr] = 0

        grouped_data[json_file_number]["abbr_count"][abbr] += 1
        grouped_data[json_file_number][filename][abbr].append(entry)

    # Print the grouped data
    for json_file_number, data in grouped_data.items():
        print(f"json_file={json_file_number}:")
        print(f"  Unique abbr: {data['unique_abbr']}")
        print("  Abbr counts:")
        for abbr, count in data["abbr_count"].items():
            print(f"    {abbr}: {count}")
        print()

    pickle_file = "processed_data.pkl"
    if os.path.exists(pickle_file):
        logging.info(f"Loading data from existing pickle file: {pickle_file}")
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f)
    # Convert pickle_data to DataFrame
    pickle_df = pd.DataFrame(pickle_data)

    # count tags per image
    data_for_df=[]
    for filename in os.listdir(concept_folder):
        if not filename.endswith(".jpg"):  # Ensure you're only processing .jpg files
            continue
        if "type-neg" in filename:
            continue
        rest, filename = filename.split("_filename-")
        parts=rest.split("_")
        row = {f'{part.split("-")[0]}': part.split("-")[1] for i, part in enumerate(parts)}
        row["filename"]=filename
        data_for_df.append(row)

    images_df = pd.DataFrame(data_for_df)

    folders = []
    data_for_df=[]
    for item in os.listdir(concept_folder):
        item_path = os.path.join(concept_folder, item)
        if os.path.isdir(item_path):
            folders.append(item)
    for folder in folders:
        if "negative" in folder:
            continue
        parts = folder.split("_")
        row = {'foldername': folder}  # Add foldername to the row
        # Split folder name by underscores and process parts
        parts = folder.split("_")
        for part in parts:
            key, value = part.split("-", 1)  # Split into key-value pair
            row[key] = value  # Add to row dictionary
        data_for_df.append(row)

    folder_df = pd.DataFrame(data_for_df)

    # Extract relevant columns
    relevant_pickle_df = pickle_df[['filename', 'region_type', 'expansion_factor', 'annotation_form', 'polygon_nr', 'json_file']]
    relevant_pickle_df = relevant_pickle_df.rename(columns={'region_type': 'abbr', 'polygon_nr': 'nr', 'json_file': 'json', 'expansion_factor': 'exp','annotation_form':'form'})

    relevant_pickle_df['df'] = "pickle"
    images_df['df'] = 'images'
    folder_df['df'] = 'folder'

    # Concatenate the DataFrames vertically
    combined_df = pd.concat([relevant_pickle_df, images_df,folder_df], ignore_index=True)
    # Convert 'StringColumn' to float using astype()
    combined_df['exp'] = pd.to_numeric(combined_df['exp'], errors='coerce')
    combined_df.to_csv('concept_df.csv', index=False)

    relevant_pickle_df = combined_df[combined_df['df'] == 'pickle']
    images_df = combined_df[combined_df['df'] == 'images']
    folder_df = combined_df[combined_df['df'] == 'folder']

    print('\n')
    print('#### check if file has all parameter sets ###')
    print('\n')

    #count
    # Filter out rows where type is 'neg' or form is 'original'
    filtered_main = images_df[(images_df['type'] != 'neg') & (images_df['form'] != 'original')]
    grouped = filtered_main.groupby(['filename', 'nr', 'json','abbr'])

    # Separate groups for 'type' = 'neg' and 'form' = 'original'
    filtered_neg = images_df[images_df['type'] == 'neg']
    filtered_original_form = images_df[images_df['form'] == 'original']

    # Step 4: Check for the presence of specific variations within each group
    # required_exp = {'1', '1.5', '2', '5'}
    required_exp = {1,1.5,2,5} #todo get values dynamically?
    required_form = {'polygon'}
    required_resize = {'true', 'false'}
    required_bg = {'original', 'synth'}
    # Generate all possible combinations of the required variations
    required_combinations_polygon = list(itertools.product(required_exp, required_form, required_resize, required_bg))
    #required_exp = {'1', '1.5', '2', '5'}
    required_exp = {1,1.5,2,5}
    required_form = {'square'}
    required_resize = {'true', 'false'}
    required_bg = {'original'}
    required_combinations_square = list(itertools.product(required_exp, required_form, required_resize,required_bg))
    required_combinations_images=required_combinations_polygon+required_combinations_square

    # Step 4: Check for the presence of specific variations within each group
    def check_combinations(df):
        existing_combinations = set(zip(df['exp'], df['form'], df['resize'], df['bg']))
        return all(comb in existing_combinations for comb in required_combinations_images)

    # Step 5: Apply the check to each group and collect results
    results = {}
    for idx,(name, group) in enumerate(grouped):
        results[name] = check_combinations(group)

    # Step 6: Separate the groups that have all combinations and those that do not
    groups_with_all_combinations = {name: present for name, present in results.items() if present}
    print(f"Number of groups with all combinations: {len(groups_with_all_combinations)}")
    groups_missing_combinations = {name: present for name, present in results.items() if not present}
    # Check if groups_missing_combinations is empty
    if not groups_missing_combinations:
        print("No groups are missing combinations.")
    else:
        print(f"Number of groups missing combinations: {len(groups_missing_combinations)}")

    # Extract unique filenames from groups_with_all_combinations
    unique_filenames = set(name[0] for name in groups_with_all_combinations)

    # Check if each unique filename has a corresponding filename in filtered_original_form
    matching_filenames = unique_filenames.intersection(set(filtered_original_form['filename'].unique()))

    # Ensure no left-outs: all filenames in groups_with_all_combinations should be in matching_filenames
    all_matched = unique_filenames == matching_filenames

    print("\nAll groups matched an original:", all_matched)

    # Extract unique filenames from groups_with_all_combinations
    unique_matched_filenames = set(name[0] for name in groups_with_all_combinations)
    unique_orginal_filenames = set(filtered_original_form['filename'].unique())

    def filter_files(concept_folder, unique_matched_filenames, unique_original_filenames):
        all_files_in_directory = [f for f in os.listdir(concept_folder) if
                                  os.path.isfile(os.path.join(concept_folder, f))]

        # Filter out files that contain any of the unique matched or original filenames
        filtered_files = [
            file for file in all_files_in_directory
            if not any(umf in file for umf in unique_matched_filenames) and
               not any(uof in file for uof in unique_original_filenames) and
           "form-na" not in file
        ]

        return filtered_files

    # Assuming concept_folder, unique_matched_filenames, and unique_original_filenames are already defined
    filtered_files = filter_files(concept_folder, unique_matched_filenames, unique_orginal_filenames)

    print(f"Files that are too much?: {filtered_files}")

    print('\n')
    print('#### check if every image has corresponding pickle ###')
    print('\n')

    # Filter pickle_df based on criteria
    filtered_pickle_df = relevant_pickle_df[(relevant_pickle_df['form'] == 'polygon') & (relevant_pickle_df['exp'] == 1)]

    # Define required combinations
    required_combinations_pickle = filtered_pickle_df[['form', 'exp', 'abbr', 'filename']].values.tolist()

    # Filter images_df to only include rows where form='polygon' and exp=1
    filtered_images_df = images_df[(images_df['form'] == 'polygon') & (images_df['exp'] == 1)]

    # Convert filtered images_df columns to tuples for comparison
    existing_combinations = filtered_images_df[['form', 'exp', 'abbr', 'filename']].apply(tuple, axis=1).tolist()

    # Separate combinations into two lists: existent and non-existent in images_df
    existent_combinations = [comb for comb in required_combinations_pickle if tuple(comb) in existing_combinations]
    print(f"Length of existing combinations: {len(existent_combinations)}")
    non_existent_combinations = [comb for comb in required_combinations_pickle if tuple(comb) not in existing_combinations]
    print(f"Length of non-existing combinations: {len(non_existent_combinations)}")

    # Print the result
    if not non_existent_combinations:
        print("All combinations in pickle_df have a corresponding element in images_df.")
    else:
        print("Not all combinations in pickle_df have a corresponding element in images_df.")
        for comb in non_existent_combinations:
            print(comb)

    print('\n')
    print('#### check if every folder has corresponding images ####')
    print('\n')

    # Filter folder_df to exclude rows with 'neg' or 'form' as 'original'
    specific_foldername="type-na_abbr-na_exp-na_form-original_resize-na_bg-na_json-na"
    filtered_folder_df = folder_df[(folder_df['type'] != 'neg') & (folder_df['form'] != 'original') &
    (folder_df['foldername'] != specific_foldername)]

    # Compare lengths
    expected_length = len(groups_with_all_combinations)
    actual_length = len(filtered_folder_df)

    if actual_length == expected_length:
        print("The length of filtered_folder_df matches the expected length.")
    else:
        print(
            f"The length of filtered_folder_df ({actual_length}) does not match the expected length ({expected_length}).")  #todo find out where that difference comes from!!!

    grouped = filtered_folder_df.groupby(['json', 'abbr'])

    # Step 5: Apply the check to each group and collect results
    results = {}
    for name, group in grouped:
        results[name] = check_combinations(group)

    # Step 6: Separate the groups that have all combinations and those that do not
    groups_with_all_combinations = {name: present for name, present in results.items() if present}
    print(f"Number of groups with all combinations: {len(groups_with_all_combinations)}")
    groups_missing_combinations = {name: present for name, present in results.items() if not present}
    # Check if groups_missing_combinations is empty
    if not groups_missing_combinations:
        print("No groups are missing combinations.")
    else:
        print(f"Number of groups missing combinations: {len(groups_missing_combinations)}")




    return

if __name__ == "__main__": #todo
    # Initialize logging
    setup_logging(log_dir='concept_prep')
    logging.info(f"Script execution started")

    config = load_config()
    concept_folder = config.get("concept_folder", "/home/fkraehenbuehl/projects/CaptumTCAV/data/concepts/")
    force_overwrite = config.get("force_overwrite", False)
    json_files = config.get("json_files", "project_vgg_annotation_1.json")


    #only required for setup (or re-setup)
    get_concepts(force_overwrite)# todo uncomment
    get_negative_concepts(force_overwrite)# todo uncomment
    # # TCAV concepts rely on folder-structure
    prepare_concept_folders(concept_folder,force_overwrite)#todo uncomment  json_files = config.get("json_files", "project_vgg_annotation_1.json")
    generate_report(concept_folder, json_files)

