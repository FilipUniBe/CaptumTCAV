import json
import os
import random
import re
import sys

import cv2
import numpy as np
import pandas as pd
from shapely import Polygon



def clean_up_data(data):
    # Filter data to keep only entries with "patient" in the filename
    cleaned_data = [entry for entry in data if "patient" in entry["filename"]]

    # Logging the number of removed entries
    removed_entries_count = len(data) - len(cleaned_data)

    return cleaned_data
def extract_data(json_files):
    """
        Brief description of what the function does.

        Detailed description of the function. This can include explanations of the parameters,
        the return value, and any exceptions that might be raised.

        Parameters:
        json_files (list(string)): A list of strings that are the names of the annotation files to consider

        Attention:
        ONLY POLYGONAL ANNOTATIONS ARE CONSIDERED!

        Returns:
        extracted_data: A dict with every polygon - per every filename - per every region_type

        Example:
        "filename": filename,
                        "all_points_x": all_points_x,
                        "all_points_y": all_points_y,
                        "region_type": region_type,
                        "expansion_factor": 1,
                        "annotation_form": "polygon",
                        "polygon_nr": polygon_nr,
                        "json_file": json_file_number,
                        "polygon_identifier":polygon_identifier
        """

    # initiate variables for loops
    extracted_data = []
    unique_polygons = set()
    polygon_counts = {}

    # loop trough given json files
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)

        # loop through json entries
        for entry_key, entry_value in data["_via_img_metadata"].items():
            filename = entry_value["filename"]
            regions = entry_value["regions"]

            # loop trough region_types
            for region in regions:

                # only consider polygonal shapes
                if region["shape_attributes"]["name"] != "polygon":
                    continue

                # extract data
                all_points_x = region["shape_attributes"]["all_points_x"]
                all_points_y = region["shape_attributes"]["all_points_y"]
                region_type = region["region_attributes"]["type"]

                # count polygon if multiple per file per region-type
                filename_and_regiontype = f'{filename}-{region_type}'
                if filename_and_regiontype not in polygon_counts:
                    polygon_counts[filename_and_regiontype]=0

                # Create original polygon
                polygon_points = list(zip(all_points_x, all_points_y))
                original_polygon = Polygon(polygon_points)

                # Create a unique identifier for the polygon
                polygon_identifier = tuple(polygon_points)

                # skip invalid polygons
                if not original_polygon.is_valid:
                    continue

                # Skip duplicates
                if polygon_identifier in unique_polygons:
                    continue
                unique_polygons.add(polygon_identifier)

                # count polygons if multiple per file per region-type
                polygon_counts[filename_and_regiontype] += 1
                polygon_nr = f"{polygon_counts[filename_and_regiontype]:03}"  # Zero-padded to three digits

                # get json number
                match = re.search(r'(\d+)\.json$', json_file)
                if match:
                    json_file_number = match.group(1)
                else:
                    sys.exit(1)

                # create dict entry
                extracted_data.append({
                    "filename": filename,
                    "all_points_x": all_points_x,
                    "all_points_y": all_points_y,
                    "region_type": region_type,
                    "expansion_factor": 1,
                    "annotation_form": "polygon",
                    "polygon_nr": polygon_nr,
                    "json_file": json_file_number,
                    "polygon_identifier":polygon_identifier
                })

    return extracted_data

def process_dataframe(df):
    # Find the start of columns to consider
    start_col_index = df.columns.get_loc(filter_column)

    # Get columns to keep (those in specified_columns)
    columns_to_keep = [col for col in df.columns if col in specified_columns]

    # Filter rows where -1 appears in the specified columns
    mask_invalid = df[columns_to_keep].isin([-1]).any(axis=1)
    df = df[~mask_invalid]

    # Drop rows where only zeros appear in the specified columns
    mask_zero = (df[columns_to_keep] == 0).all(axis=1)
    df = df[~mask_zero]

    # Fill empty fields with 0
    df.fillna(0, inplace=True)

    # Drop columns from "No Finding" onward that are not in specified_columns
    columns_to_drop = [col for col in df.columns[start_col_index:] if col not in specified_columns]

    # Update the DataFrame in place
    df.drop(columns=columns_to_drop, inplace=True)

    # Drop rows where 'Path' ends with 'lateral.jpg'
    df = df[~df['Path'].str.endswith('lateral.jpg')]

    df = df.reset_index(drop=True)

    return df

def rebalance_dataframe(df, labels):

    # Get the counts of positive examples for each label
    label_counts = {label: df[df[label] == 1].shape[0] for label in labels}

    # Find the minimum number of positive examples across all labels
    min_count = min(label_counts.values())

    # Sample data for each label to have equal number of positive examples
    sampled_dfs = [
        df[df[label] == 1].sample(min_count, replace=False, random_state=42)
        for label in labels
    ]

    # Concatenate sampled data
    rebalanced_df = pd.concat(sampled_dfs)

    # Shuffle the DataFrame and reset the index
    rebalanced_df = rebalanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return rebalanced_df

def process_dataframe_new(df):
    # Find the start of columns to consider
    start_col_index = df.columns.get_loc(filter_column)

    # Filter rows where -1 appears in any column
    mask_invalid = df.isin([-1]).any(axis=1)
    df = df[~mask_invalid]

    # Fill empty fields with 0
    df.fillna(0, inplace=True)

    # Drop rows where 'Path' ends with 'lateral.jpg'
    df = df[~df['Path'].str.endswith('lateral.jpg')]

    df = df.reset_index(drop=True)

    return df

def save_image_patches(df, output_subdir, image_base_dir, all_points):

    # Ensure the output directory exists
    output_base_dir = r"/home/fkraehenbuehl/projects/CaptumTCAV/data/concepts"
    output_dir=os.path.join(output_base_dir,output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through the images in the DataFrame and the cropping points
    for idx, (image_path, (x_points, y_points)) in enumerate(zip(df['Path'], all_points)):

        # Load the image using OpenCV (or use an alternative if needed)
        full_image_path = os.path.join(image_base_dir, image_path)
        image = cv2.imread(full_image_path)

        if image is None:
            print(f"Error loading image {full_image_path}")
            continue

        # Convert x_points and y_points to a numpy array of shape (n, 2) for OpenCV
        polygon = np.array(list(zip(x_points, y_points)), dtype=np.int32)

        # Create a mask of the same size as the image, initialized to zero (black)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Fill the polygon with white (255) on the mask
        cv2.fillPoly(mask, [polygon], 255)

        # Use the mask to extract the polygonal area from the image
        cropped_image = cv2.bitwise_and(image, image, mask=mask)

        # Extract the bounding rectangle of the polygon to save just the relevant part
        x, y, w, h = cv2.boundingRect(polygon)
        cropped_image = cropped_image[y:y + h, x:x + w]

        if cropped_image is None or cropped_image.size == 0:
            print(f"Error: Cropped image is empty for {full_image_path}")
            continue

        # Save the cropped patch with a unique name
        #healthy_train_filename-patient53415-study1-view1_frontal.jpg
        relative_image_path = full_image_path.replace('/home/fkraehenbuehl/projects/CheXpert-v1.0/',"")
        relative_image_path = relative_image_path.replace(".jpg","")
        relative_image_path = relative_image_path.replace("/","-")#test/patient64763/study1/view1_frontal.jpg

        #/home/fkraehenbuehl/projects/CheXpert-v1.0/test/patient65026/study1/view1_frontal.jpg
        #/home/fkraehenbuehl/projects/CheXpert-v1.0/test/patient64835/study1/view1_frontal.jpg
        #/home/fkraehenbuehl/projects/CheXpert-v1.0/test/patient64793/study1/view1_frontal.jpg
        #/home/fkraehenbuehl/projects/CheXpert-v1.0/test/patient64969/study1/view1_frontal.jpg #todo investigate cropping error

        patch_filename=f'patch_{relative_image_path}.jpg'
        patch_path = os.path.join(output_dir, patch_filename)
        cv2.imwrite(patch_path, cropped_image)

        print(f"Saved patch {patch_filename} at {patch_path}")
    return

# Define the specified columns and the column to filter
specified_columns = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
filter_column = "No Finding"

json_files = [
    "project_vgg_annotation_1.json",
    "project_vgg_annotation_2.json",
    "project_vgg_annotation_3.json",
    "project_vgg_annotation_4.json",
    "project_vgg_annotation_5.json",
    "project_vgg_annotation_6.json",
    "project_vgg_annotation_7.json"
]
json_folder="/home/fkraehenbuehl/projects/CaptumTCAV/concept_preparation"


json_files_paths=[os.path.join(json_folder, json_file) for json_file in json_files]
new_data = extract_data(json_files_paths)
pickle_data = clean_up_data(new_data)

all_points = [(entry["all_points_x"], entry["all_points_y"]) for entry in pickle_data]

csv_file_test = r"/home/fkraehenbuehl/projects/CheXpert-v1.0/valid.csv"#test.csv
image_base_dir = r"/home/fkraehenbuehl/projects/CheXpert-v1.0/"

test_df = pd.read_csv(csv_file_test)
test_df = process_dataframe_new(test_df) #todo actually shouldn't matter if we have more pathologies than in training set?
# Define labels to rebalance
labels = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"] #todo from config
# Rebalance DataFrame based on the specified labels
rebalanced_df = rebalance_dataframe(test_df, labels)

# Filter DataFrame based on criteria
filtered_df = test_df[test_df["No Finding"] == 1]

random_df = rebalanced_df.sample(frac=1).reset_index(drop=True)  # Shuffle the DataFrame
healthy_df = filtered_df.sample(frac=1).reset_index(drop=True)  # Shuffle the DataFrame

# Assuming all_points is the list of cropping points corresponding to the images
# Process and save patches for random_df (random patches)
save_image_patches(random_df, "random_patches_valid", image_base_dir, all_points)

# Process and save patches for healthy_df (healthy patches)
save_image_patches(healthy_df, "healthy_patches_valid", image_base_dir, all_points)

