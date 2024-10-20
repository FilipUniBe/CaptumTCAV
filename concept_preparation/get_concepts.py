import itertools
import pickle
import re
import shutil
import sys
import logging

import cv2
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import yaml
from matplotlib import image as mpimg
from shapely.geometry import Polygon, box
from sklearn.cluster import KMeans
from tqdm import tqdm

from config import load_config
from operator import itemgetter
from PIL import Image, ImageDraw

from shapely.validation import explain_validity


def calculate_average_polygon(polygons):
    # Initialize variables to store sum of centroids
    num_polygons = len(polygons)

    if num_polygons == 0:
        return None, None, None, None, None, None, None, None, None, None

    # Lists to store centroid coordinates, areas, and aspect ratios
    centroids_x = []
    centroids_y = []
    areas = []
    aspect_ratios = []
    num_vertices_list = []

    for polygon in polygons:
        # Calculate centroid of the polygon
        centroid_x, centroid_y = polygon.centroid.xy
        centroids_x.append(centroid_x[0])
        centroids_y.append(centroid_y[0])

        # Convert Shapely Polygon to NumPy array of vertices
        vertices = np.array(polygon.exterior.coords)
        num_vertices_list.append(len(vertices))

        # Calculate area
        try:
            vertices = vertices.reshape((-1, 1, 2)).astype(np.int32)
            area = cv2.contourArea(vertices)
            areas.append(area)
        except Exception as e:
            print("Error calculating area:", e)
            areas.append(0)

        # Calculate bounding box to get aspect ratio
        bbox = polygon.bounds
        aspect_ratio = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])
        aspect_ratios.append(aspect_ratio)

    # Combine centroids into an array for clustering
    centroids = np.array(list(zip(centroids_x, centroids_y)))

    # Perform K-Means clustering with 2 clusters
    kmeans = KMeans(n_clusters=2)
    clusters = kmeans.fit_predict(centroids)

    # Split data into two clusters
    cluster1 = [i for i in range(num_polygons) if clusters[i] == 0]
    cluster2 = [i for i in range(num_polygons) if clusters[i] == 1]

    def calc_stats(indices, data):
        selected_data = [data[i] for i in indices]
        mean = np.mean(selected_data)
        std = np.std(selected_data)
        return mean, std

    # Calculate means and standard deviations for both clusters
    avg_x1, std_x1 = calc_stats(cluster1, centroids_x)
    avg_y1, std_y1 = calc_stats(cluster1, centroids_y)
    avg_area1, std_area1 = calc_stats(cluster1, areas)
    avg_aspect_ratio1, std_aspect_ratio1 = calc_stats(cluster1, aspect_ratios)
    avg_vertices1, std_vertices1 = calc_stats(cluster1, num_vertices_list)

    avg_x2, std_x2 = calc_stats(cluster2, centroids_x)
    avg_y2, std_y2 = calc_stats(cluster2, centroids_y)
    avg_area2, std_area2 = calc_stats(cluster2, areas)
    avg_aspect_ratio2, std_aspect_ratio2 = calc_stats(cluster2, aspect_ratios)
    avg_vertices2, std_vertices2 = calc_stats(cluster2, num_vertices_list)

    # Combine results into dictionaries for easier access
    stats_cluster1 = {
        'avg_x': avg_x1, 'std_x': std_x1, 'avg_y': avg_y1, 'std_y': std_y1,
        'avg_area': avg_area1, 'std_area': std_area1, 'avg_aspect_ratio': avg_aspect_ratio1, 'std_aspect_ratio': std_aspect_ratio1,
        'avg_vertices': avg_vertices1, 'std_vertices': std_vertices1
    }

    stats_cluster2 = {
        'avg_x': avg_x2, 'std_x': std_x2, 'avg_y': avg_y2, 'std_y': std_y2,
        'avg_area': avg_area2, 'std_area': std_area2, 'avg_aspect_ratio': avg_aspect_ratio2, 'std_aspect_ratio': std_aspect_ratio2,
        'avg_vertices': avg_vertices2, 'std_vertices': std_vertices2
    }

    def combine_cluster_stats(stats_cluster1, stats_cluster2):
        combined_stats = {}

        # Calculate mean and std for delta_x and delta_y
        combined_stats['mean_delta_x'] = (stats_cluster1['avg_x'] + stats_cluster2['avg_x']) / 2
        combined_stats['std_delta_x'] = (stats_cluster1['std_x'] + stats_cluster2['std_x']) / 2
        combined_stats['mean_delta_y'] = (stats_cluster1['avg_y'] + stats_cluster2['avg_y']) / 2
        combined_stats['std_delta_y'] = (stats_cluster1['std_y'] + stats_cluster2['std_y']) / 2

        # For positions, combine directly from centroids
        combined_stats['mean_x_position'] = (stats_cluster1['avg_x'] + stats_cluster2['avg_x']) / 2
        combined_stats['std_x_position'] = (stats_cluster1['std_x'] + stats_cluster2['std_x']) / 2
        combined_stats['mean_y_position'] = (stats_cluster1['avg_y'] + stats_cluster2['avg_y']) / 2
        combined_stats['std_y_position'] = (stats_cluster1['std_y'] + stats_cluster2['std_y']) / 2

        # Total points
        combined_stats['total_points'] = len(polygons)

        return combined_stats

    stats_combined = combine_cluster_stats(stats_cluster1, stats_cluster2)

    return stats_cluster1, stats_cluster2,stats_combined

def annotation_stats(json_files,image_width,image_height):

    all_croppings_data = []
    for json_file in json_files:
        croppings_data = extract_data(json_file)
        all_croppings_data.extend(croppings_data)


    sorted_data = sorted(all_croppings_data, key=itemgetter('region_type'))
    grouped_data = [(concept, list(group)) for concept, group in itertools.groupby(sorted_data, key=itemgetter('region_type'))]



    # Example usage
    concept_polygons = {}
    # Initialize the dictionary
    result_dict = {}

    for i,(concept, group) in enumerate(grouped_data):

        delta_x_values = []
        delta_y_values = []
        x_positions = []
        y_positions = []
        total_points = 0
        polygons = []

        for entry in group:
            delta_x = max(entry["all_points_x"]) - min(entry["all_points_x"])
            delta_y = max(entry["all_points_y"]) - min(entry["all_points_y"])
            delta_x_values.append(delta_x)
            delta_y_values.append(delta_y)
            x_position = sum(entry["all_points_x"]) / len(entry["all_points_x"])
            y_position = sum(entry["all_points_y"]) / len(entry["all_points_y"])
            x_positions.append(x_position)
            y_positions.append(y_position)
            total_points += len(entry["all_points_x"])

            # Create original polygon
            polygon_points = list(zip(entry["all_points_x"], entry["all_points_y"]))
            polygon = Polygon(polygon_points)
            polygons.append(polygon)

        # Calculate average polygon
        stats_cluster1, stats_cluster2, stats_combined = calculate_average_polygon(polygons)


        # Save polygons to a file
        if concept not in concept_polygons:
            concept_polygons[concept] = []
        concept_polygons[concept]=polygons

        def add_to_dict(a, stats_cluster1, stats_cluster2, stats_combined):
            if a not in result_dict:
                result_dict[a] = {}

            result_dict[a]["stats_cluster1"] = stats_cluster1
            result_dict[a]["stats_cluster2"] = stats_cluster2
            result_dict[a]["stats_combined"] = stats_combined

            return result_dict

        result_dict = add_to_dict(concept, stats_cluster1, stats_cluster2, stats_combined)

    plot_ensemlbe_plot_for_distro(delta_x_values,delta_y_values,x_positions,y_positions,grouped_data,image_width,image_height)

    return result_dict,concept_polygons
def save_polygons(concept_polygons, filename):
    with open(filename, 'wb') as f:
        pickle.dump(concept_polygons, f)

def plot_ensemlbe_plot_for_distro(delta_x_values,delta_y_values,x_positions,y_positions,grouped_data,image_width,image_height):
    delta_combo_values = [dx * dy for dx, dy in zip(delta_x_values, delta_y_values)]

    num_concepts= len(grouped_data)

    fig, axes = plt.subplots(2, num_concepts+1, figsize=(15, 5))
    plt.suptitle("Concept Distribution (position normalized to image H*W, size to 1)")

    # Initialize lists to store total normalized values
    total_norm_x = [0] * len(delta_x_values)
    total_norm_y = [0] * len(delta_y_values)
    total_norm_x_pos = [0] * len(x_positions)
    total_norm_y_pos = [0] * len(y_positions)
    for i,(concept, group) in enumerate(grouped_data):

        norm_x = [float(i) / sum(delta_x_values) for i in delta_x_values]
        norm_y = [float(i) / sum(delta_y_values) for i in delta_y_values]
        norm_z = [float(i) / sum(delta_combo_values) for i in delta_combo_values]

        norm_x_pos = [float(i) / image_width for i in x_positions]
        norm_y_pos = [float(i) / image_height for i in y_positions]

        # Update total normalized values
        total_norm_x = [total_norm_x[i] + norm_x[i] for i in range(len(delta_x_values))]
        total_norm_y = [total_norm_y[i] + norm_y[i] for i in range(len(delta_y_values))]
        total_norm_x_pos = [total_norm_x_pos[i] + norm_x_pos[i] for i in range(len(x_positions))]
        total_norm_y_pos = [total_norm_y_pos[i] + norm_y_pos[i] for i in range(len(y_positions))]

        ax_position = axes[0, i]
        ax_size = axes[1, i]

        # Plot histograms
        plot_my_histogram(ax_size, norm_x, concept, 'Size', 'Frequency', 'r', 'X Size')
        plot_my_histogram(ax_size, norm_y, concept, 'Size', 'Frequency', 'g', 'Y Size')
        plot_my_histogram(ax_size, norm_z, concept, 'Size', 'Frequency', 'b', 'Area Size')
        plot_my_histogram(ax_position, norm_x_pos, concept, 'Position', 'Frequency', 'r', 'X Position')
        plot_my_histogram(ax_position, norm_y_pos, concept, 'Position', 'Frequency', 'b', 'Y Position')

    ax_position = axes[0, i+1]
    ax_size = axes[1, i+1]
    # Plot histograms
    plot_my_histogram(ax_size, total_norm_x, "total", 'Size', 'Frequency', 'r', 'X Size')
    plot_my_histogram(ax_size, total_norm_y, "total", 'Size', 'Frequency', 'g', 'Y Size')
    plot_my_histogram(ax_position, total_norm_x_pos, "total", 'Position', 'Frequency', 'r', 'X Position')
    plot_my_histogram(ax_position, total_norm_y_pos, "total", 'Position', 'Frequency', 'b', 'Y Position')



    plt.savefig(f'figures/Ensemble Plot for polygon distributions')  # Save the plot
    plt.close()
    print("printed Ensemble")



def plot_my_histogram(ax, data, title, xlabel, ylabel, color,label):
    ax.hist(data, bins=20, density=True, alpha=0.6, color=color,label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper right')

def extract_attribute_translations(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    translations = {}
    attributes = data["_via_attributes"]["region"]
    for abbreviation, name_mapping in attributes.items():
        if "options" in name_mapping:
            options = name_mapping["options"]
            for abbreviation, name in options.items():
                translations[abbreviation] = name

    return translations
def filename_to_path(filename):
    parts = filename.split('-')
    patient = parts[0]
    study = parts[1]
    name = parts[2]
    filepath = os.path.join(patient,study,name)
    return filepath

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
                    logging.warning(f"Invalid original polygon: {explain_validity(original_polygon)}")
                    logging.warning(f"polygon_identifier : {polygon_identifier}")
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
                    logging.error(f"No number found in json file name: {json_file}")
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


def expand_polygon(polygon, factor):
    centroid = polygon.centroid
    expanded_coords = []
    for point in polygon.exterior.coords:
        direction_vector = np.array(point) - np.array(centroid)
        expanded_point = np.array(centroid) + direction_vector * factor
        expanded_coords.append(tuple(expanded_point))
    return Polygon(expanded_coords)

def clip_polygon(polygon, width, height):
    if not polygon.is_valid:
        reason = explain_validity(polygon)
        print(f"Invalid polygon: {reason}")
        # You can try to fix the polygon, or handle the error appropriately
        return None  # Or raise an exception
    image_box = box(0, 0, width, height)
    clipped_polygon = polygon.intersection(image_box)
    return clipped_polygon

def extract_coordinates(geometry):
    """
    Extract coordinates from a Polygon or MultiPolygon.
    Returns a list of x and y coordinates.
    """
    all_points_x = []
    all_points_y = []

    # If geometry is a Polygon, treat it as a list of one Polygon
    if isinstance(geometry, Polygon):
        geometry = [geometry]

    # Iterate over each Polygon in the MultiPolygon or list of one Polygon
    for polygon in geometry:
        exterior_coords = polygon.exterior.coords.xy
        all_points_x.extend(exterior_coords[0])
        all_points_y.extend(exterior_coords[1])

    return all_points_x, all_points_y


def extract_section(image_path, bounding_box):
    """
    Extracts a section of an image specified by the bounding box.

    Args:
    - image_path (str): Path to the image file.
    - bounding_box (dict): Dictionary containing the bounding box coordinates: {"x": x, "y": y, "width": width, "height": height}.

    Returns:
    - section (numpy.ndarray): Extracted section of the image.
    """
    # Read the image
    image = cv2.imread(image_path)

    # Extract the bounding box coordinates
    x, y, width, height = bounding_box["x"], bounding_box["y"], bounding_box["width"], bounding_box["height"]

    # Extract the specified section
    section = image[y:y+height, x:x+width]

    return section



def resize_images_in_directory(input_dir, force_overwrite,new_size):
    """
    Resize all JPEG images in a directory.

    Parameters:
        input_dir (str): Path to the input directory containing JPEG images.
        output_dir (str): Path to save the resized images.
        new_size (tuple): Tuple containing the new width and height of the images.
    """

    # Iterate over files in the input directory
    filenamelist=[]
    for filename in os.listdir(input_dir):
        if not filename.endswith(".jpg") or filename.endswith(".jpeg"):
            continue
        if "form-original" in filename:
            continue
        filenamelist.append(filename)
    for filename in tqdm(filenamelist, desc="Resize and Save Images"):

        # Replace "resize_f" with "resize_t"
        new_filename = filename.replace("resize-false", "resize-true")
        save_path=os.path.join(input_dir, new_filename)
        # Check if the exact same file exists already
        if os.path.exists(save_path) and not force_overwrite:
            logging.info(f"Skipping {save_path} (already exists)")
            continue

        # Open the input image
        with Image.open(os.path.join(input_dir, filename)) as img:
            # Resize the image
            resized_img = img.resize(new_size)

            # Save the resized image to the output directory
            resized_img.save(save_path)
            logging.info(f"Image resized and saved: {new_filename}")

def construct_filename(entry, filename, type,resize,bg,annotation_form,exp=None,nr=None,json=None):
    # Define the tags
    expansion_factor = exp if exp is not None else entry.get("expansion_factor", "default_exp")
    poly_nr = nr if nr is not None else entry.get("polygon_nr", "default_exp")
    json_file = json if json is not None else entry.get("json_file", "default_exp")
    tags = {
        "type": type,
        "abbr": entry['region_type'],
        "exp": expansion_factor,
        "form": annotation_form,
        "resize": resize,
        "bg": bg,
        "nr": poly_nr,
        "json":  json_file,
        "filename": entry["filename"]
    }

    # Construct the output filename using the tags
    output_filename = "_".join([f"{key}-{value}" for key, value in tags.items()])

    return output_filename
def crop_image(data, image_folder, output_folder,force_overwrite):

    for entry in data:

        # Extract filenames
        filename = entry['filename']
        filepath = filename_to_path(filename)
        filepath = os.path.join(image_folder, filepath)

        if entry["json_file"]=="5":
            pass
        if entry["json_file"]=="7":
            pass
        if entry["json_file"]=="6":
            filepath=filepath.replace("train","test")


        output_filename=construct_filename(entry, filename, type = "pos",resize="false",bg="original",annotation_form=entry["annotation_form"]) #todo replace with DB-system for scalability
        output_filepath = os.path.join(output_folder, output_filename)



        # Check if resulting image exists
        if os.path.exists(output_filepath) and not force_overwrite:
            logging.info(f"Skipping {output_filepath} (already exists)")
            continue

        # Check if uncropped image exists
        if not os.path.exists(filepath):
            logging.warning(f"Image file {filename} not found.")
            continue

        # Load the image
        image = cv2.imread(filepath)
        if image is None:
            logging.warning(f"Failed to load image: {filename}")
            continue

        # Get the polygon coordinates
        points_x = entry["all_points_x"]
        points_y = entry["all_points_y"]

        # Create a mask from the polygon
        mask = np.zeros_like(image)
        pts = np.array(list(zip(points_x, points_y)), dtype=np.int32)
        cv2.fillPoly(mask, [pts], (255, 255, 255))

        # Apply the mask to the original image
        masked_image = cv2.bitwise_and(image, mask)

        # Find the bounding box of the masked region
        x, y, w, h = cv2.boundingRect(pts)

        # Crop the original image using the bounding box
        cropped_image = masked_image[y:y + h, x:x + w]

        # Save the cropped image
        cv2.imwrite(output_filepath, cropped_image)
        logging.info(f"Image cropped and saved: {output_filepath}")



def increase_contrast(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Compute min and max pixel values
    min_val, max_val, _, _ = cv2.minMaxLoc(gray)
    # Linearly scale pixel values to the full dynamic range [0, 255]
    contrast_image = ((gray - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    # Convert back to BGR color space
    contrast_image = cv2.cvtColor(contrast_image, cv2.COLOR_GRAY2BGR)
    return contrast_image



def print_hardcoded_bg_patch():
    background_img_section = "/home/fkraehenbuehl/projects/CheXpert-v1.0/train/patient00199/study1/view1_frontal.jpg"  # fixme bg simulation seems silly to me #todo put to config #no must be curated

    background_img_box = {"x": 64, "y": 3, "width": 320, "height": 210}
    background_img = extract_section(background_img_section, background_img_box)
    cv2.imwrite("figures/background_section.jpg", background_img)
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(background_img, cv2.COLOR_BGR2GRAY)

    # Enhance the color difference in the patch
    enhanced_diff_img = increase_contrast(background_img)

    # Draw a red rectangle on the original image
    start_point = (background_img_box['x'], background_img_box['y'])
    end_point = (
        background_img_box['x'] + background_img_box['width'], background_img_box['y'] + background_img_box['height'])
    color = (0, 0, 255)  # Red color in BGR
    thickness = 2  # Thickness of 2 px

    # Read the image
    original_img = cv2.imread(background_img_section)
    original_img_with_box = original_img.copy()
    cv2.rectangle(original_img_with_box, start_point, end_point, color, thickness)

    # Plot the original image with the red rectangle, the patch, and the histogram
    plt.figure(figsize=(15, 5))

    # Original image with red rectangle
    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(original_img_with_box, cv2.COLOR_BGR2RGB))
    plt.title('Original Image with Red Box')
    plt.axis('off')

    # Extracted patch
    plt.subplot(1, 4, 2)
    plt.imshow(cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB))
    plt.title('Extracted Patch')
    plt.axis('off')

    # Histogram
    plt.subplot(1, 4, 3)
    plt.hist(gray_image.ravel(), bins=256, range=[0, 256], color='black', alpha=0.75)
    plt.title('Grayscale Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    # Enhanced color difference
    plt.subplot(1, 4, 4)
    plt.imshow(cv2.cvtColor(enhanced_diff_img, cv2.COLOR_BGR2RGB))
    plt.title('Enhanced Color Differences')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("figures/Ensemble background_section")

def square_polygon(data, image_width, image_height):
    new_entries = []

    for entry in data:

        annotation_form="square"

        all_points_x = entry["all_points_x"]
        all_points_y = entry["all_points_y"]
        filename = entry["filename"]
        region_type = entry["region_type"]
        polygon_nr = entry["polygon_nr"]
        json_file= entry["json_file"]
        polygon_identifier= entry["polygon_identifier"]

        # Create original polygon
        polygon_points = list(zip(all_points_x, all_points_y))
        original_polygon = Polygon(polygon_points)


        # Perform specific operation (expand, square, etc.)
        processed_polygon = original_polygon.envelope

        if not processed_polygon.is_valid:
            logging.warning(f"Invalid original polygon: {explain_validity(original_polygon)}")
            logging.warning(f"polygon_identifier : {polygon_identifier}")
            continue

        # Clip the polygon
        clipped_polygon = clip_polygon(processed_polygon, image_width, image_height)
        all_points_x, all_points_y = extract_coordinates(clipped_polygon)

        new_entry = {
            "filename": filename,
            "all_points_x": all_points_x,
            "all_points_y": all_points_y,
            "region_type": region_type,
            "expansion_factor": entry['expansion_factor'],
            "annotation_form": annotation_form,
            "polygon_nr": polygon_nr,
            "json_file": json_file,
            "polygon_identifier": polygon_identifier
        }

        new_entries.append(new_entry)
    data.extend(new_entries)
    return data
def expanion_polygon(data, image_width, image_height, params):
    new_entries = []

    for entry in data:

        #skip squares, skip non-polygons
        if entry["annotation_form"] != "polygon" or entry["expansion_factor"] != 1:
            continue
        else:
            annotation_form = "polygon"

        all_points_x = entry["all_points_x"]
        all_points_y = entry["all_points_y"]
        filename = entry["filename"]
        region_type = entry["region_type"]
        polygon_nr = entry["polygon_nr"]
        json_file= entry["json_file"]
        polygon_identifier= entry["polygon_identifier"]

        # Create original polygon
        polygon_points = list(zip(all_points_x, all_points_y))
        original_polygon = Polygon(polygon_points)

        for param in params:
            # Perform specific operation (expand, square, etc.)
            processed_polygon = expand_polygon(original_polygon, param)

            if not processed_polygon.is_valid:
                logging.warning(f"Invalid original polygon: {explain_validity(original_polygon)}")
                logging.warning(f"polygon_identifier : {polygon_identifier}")
                continue

            # Clip the polygon
            clipped_polygon = clip_polygon(processed_polygon, image_width, image_height)
            all_points_x, all_points_y = extract_coordinates(clipped_polygon)

            new_entry = {
                "filename": filename,
                "all_points_x": all_points_x,
                "all_points_y": all_points_y,
                "region_type": region_type,
                "expansion_factor": param,
                "annotation_form": annotation_form,
                "polygon_nr": polygon_nr,
                "json_file": json_file,
                "polygon_identifier": polygon_identifier
            }

            new_entries.append(new_entry)
    data.extend(new_entries)
    return data

# Placeholder functions for specific operations (expand, square, etc.)


def square_operation(polygon, _): #todo param "used" to have same logic. shitty, I know
    # Placeholder for square operation logic
    return polygon.envelope

def add_uncropped_concepts(data,image_base_dir,output_folder,force_overwrite):
    listoffiles=set()
    for file in os.listdir(output_folder):
        filepath = os.path.join(output_folder, file)
        if not os.path.isfile(filepath):  # Check if it's a file
            continue
        parts=file.split("filename-")[-1]
        listoffiles.add(parts)

    for filename in listoffiles:
        original_filename=filename.replace(".jpg","")
        # Save the cropped image
        output_filename = f"type-na_abbr-na_exp-na_form-original_resize-na_bg-na_nr-na_json-na_filename-{original_filename}.jpg"
        output_filepath = os.path.join(output_folder, output_filename)
        input_filepath=filename_to_path(original_filename)
        full_input_path = os.path.join(image_base_dir,"train", input_filepath.replace("CheXpert-v1.0/", "")) #todo hard-coded for train
        full_input_path=full_input_path+".jpg"
        # Check if the exact same file exists already
        if os.path.exists(output_filepath):
            if not force_overwrite:
                logging.info(f"Skipping {output_filepath} (already exists)")
                continue


        #copy image
        shutil.copy(full_input_path,output_filepath)
        logging.info(f"Image copied to: {output_filepath}")

def add_synth_background(inverted_mask):
    pass
    return

def filter_filenames(folder_path, keywords):
    """
    Filters filenames in the specified folder that contain all the given keywords.

    :param folder_path: Path to the folder containing files.
    :param keywords: List of keywords to filter filenames by.
    :return: List of filtered filenames.
    """
    filenames = os.listdir(folder_path)
    filtered_filenames = [
        filename for filename in filenames
        if all(keyword in filename for keyword in keywords)
    ]
    return filtered_filenames


def extract_abbr(filename):
    # Define the regular expression pattern
    pattern = r'abbr-([^-]+)_exp'

    # Search for the pattern in the filename
    match = re.search(pattern, filename)

    # Extract the matching part if found
    if match:
        return match.group(1)
    else:
        print(f"No match found in filename: {filename}")
        return None

def extract_polynr(filename):
    match = re.search(r'nr-(\d+)_json-', filename)  # Search for the pattern between "nr-" and "_json-"
    if match:
        number = match.group(1)  # Extract the number
    else:
        print("No match found")
        sys.exit()  # Exit the program with an error code
    return int(number)

# Function to calculate bounding box coordinates
def calculate_bounding_box(points_x, points_y):
    min_x = min(points_x)
    max_x = max(points_x)
    min_y = min(points_y)
    max_y = max(points_y)
    return min_x, min_y, max_x, max_y
def synthetisize_background(concept_folder,data,force_overwrite): #todo tag in names wasn't such a good idea after all :'-(
    # Load the noise patch image
    noise_patch="figures/background_section.jpg"
    noise_patch_image = Image.open(noise_patch)

    #filter for only polygon images (only ones with backgrounnd)
    keywords=["polygon","resize-false"]
    only_polygon_files=filter_filenames(concept_folder, keywords)

    for filename in only_polygon_files:
        if not filename.endswith(".jpg") or filename.endswith(".jpeg"):
            continue

        type_key,abbr_key,exp_key, form_key, resize_key, bg_key, nr_key, json_file_number, original_filename = extract_components(filename)

        original_filename=original_filename+'.jpg'

        #filter data dict
        #ensure that file picked from directory is the same as in data-dict!!
        filtered_data = [
            item for item in data
            if item["filename"] == original_filename
               and item["annotation_form"] == "polygon"
               and item["polygon_nr"] == nr_key
               and item["region_type"] == abbr_key
               and float(item["expansion_factor"]) == float(exp_key) #because of possibility of decimals
            and item["json_file"] == json_file_number
        ]

        if len(filtered_data) != 1:
            logging.warning("Warning: filtered_data must have only one key-value pair.")
            continue

        for entry in filtered_data: #supposedly only polygons

            output_filename = construct_filename(entry, filename, type="pos", resize="false",
                                                 bg="synth", annotation_form=entry[
                    "annotation_form"])  # todo replace with DB-system for scalability
            output_filepath = os.path.join(concept_folder, output_filename)
            # Check if the exact same file exists already
            if os.path.exists(output_filepath) and not force_overwrite:
                logging.info(f"Skipping {output_filepath} (already exists)")
                continue

            #get all corresponding data for copy
            all_points_x = entry["all_points_x"]
            all_points_y = entry["all_points_y"]

            # Calculate bounding box coordinates
            min_x, min_y, max_x, max_y = min(all_points_x), min(all_points_y), max(all_points_x), max(all_points_y)
            offset_x=0+min_x
            offset_y=0+min_y
            points_x = [x-offset_x for x in entry["all_points_x"]]
            points_y = [y-offset_y for y in entry["all_points_y"]]
            polygon_points = list(zip(points_x, points_y))

            # load concept image
            fullfilename = os.path.join(concept_folder, filename)
            image = Image.open(fullfilename)
            image_width, image_height = image.size

            mask = Image.new('L', (image_width, image_height), 0)
            ImageDraw.Draw(mask).polygon(polygon_points, outline="white", fill="white")
            #mask.save("mask.jpg")

            # Convert noise patch to grayscale and resize it to match the image size
            noise_patch_gray = noise_patch_image.convert('L').resize((image_width, image_height))

            # Create a bright color patch to match the image size
            bright_color_patch = Image.new('RGB', (image_width, image_height), #only visualization (debug)
                                           color=(255, 0, 0))  # Using bright red for visualization

            # # Add noise to the outside of the polygon
            composite_image = Image.composite(image,noise_patch_gray,
                                           mask)

            # Save the cropped image
            composite_image.save(output_filepath)
            logging.info(f"Synthesized Background for: {output_filepath}")


    return

def extract_components(filename):
    # Define the regular expression pattern
    #pattern = r"type-(\w+)_abbr-(\w+)_exp-(\d+(\.\d+)?)_form-(\w+)_resize-(\w+)_bg-(\w+)_nr-(\d+)_patient\d+-study\d+-view\d+_frontal\.jpg"
    pattern = r"type-(\w+)_abbr-(\w+)_exp-(\d+(\.\d+)?)_form-(\w+)_resize-(\w+)_bg-(\w+)_nr-(\d+)_json-(\d+)_filename-(.+)\.jpg"

    # Match the pattern to the filename
    match = re.match(pattern, filename)

    # Extract components
    if match:
        concepttype= match.group(1)
        abbr=match.group(2)
        exp = match.group(3)
        #decimalornot=match.group(4) #not used actually #todo delte?
        form = match.group(5)
        resize=match.group(6)
        bg=match.group(7)
        nr=match.group(8)
        json_file_number = match.group(9)
        original_filename = match.group(10)
        return concepttype,abbr,exp, form, resize, bg, nr, json_file_number, original_filename
    else:
        return None
def extract_json_numbers(filenames):
    """
    Extracts numbers from filenames that match the pattern _json-#_.

    :param filenames: List of filenames to search.
    :return: List of extracted numbers.
    """
    pattern = re.compile(r"_json-(\d+)_")
    numbers = []

    for filename in filenames:
        match = pattern.search(filename)
        if match:
            numbers.append(int(match.group(1)))

    return numbers
def extract_polynr_numbers(filenames):
    """
    Extracts numbers from filenames that match the pattern _json-#_.

    :param filenames: List of filenames to search.
    :return: List of extracted numbers.
    """
    pattern = re.compile(r"_nr-(\d+)_")
    numbers = []

    for filename in filenames:
        match = pattern.search(filename)
        if match:
            numbers.append(int(match.group(1)))

    return numbers
def extract_abbr_numbers(filenames):
    """
    Extracts numbers from filenames that match the pattern abbr-###.

    :param filenames: List of filenames to search.
    :return: List of extracted numbers.
    """
    pattern = re.compile(r"abbr-(\w+)_")
    abbr_strings = []

    for filename in filenames:
        match = pattern.search(filename)
        if match:
            abbr_strings.append(match.group(1))

    return abbr_strings
def visualize_patient(concept_folder,random_filename,force_overwrite):
    ###visual test####
    # random_filename to match
    type_key, abbr_key, exp_key, form_key, resize_key, bg_key, nr_key, json_file_number, original_filename = extract_components(
        random_filename)  # todo later

    # filter for only polygon images (only ones with backgrounnd)
    keywords = [original_filename]
    only_filtered_files = filter_filenames(concept_folder, keywords)

    json_numbers = extract_json_numbers(only_filtered_files)
    unique_json_numbers=set(json_numbers)
    poly_numbers = extract_polynr_numbers(only_filtered_files)
    formatted_poly_numbers = [f"{num:03}" for num in poly_numbers]
    unique_poly_numbers = set(formatted_poly_numbers)
    abbr_strings  = extract_abbr_numbers(only_filtered_files)
    unique_abbr_strings = set(abbr_strings)

    for element in only_filtered_files:
        if "form-original" in element:
            originalfile=element

    for json_number in unique_json_numbers:
        for abbr_string in unique_abbr_strings:
            for poly_number in unique_poly_numbers:

                plotname=f'json-{json_number}-abbr-{abbr_string}-polynr-{poly_number}-{original_filename}'

                # Construct the full path to the file
                file_path = os.path.join("figures", f"Concept_variations_{plotname}.png")

                if os.path.exists(file_path) and not force_overwrite:
                    logging.info(f" skipped figures/Concept_variations_{plotname}")
                    continue

                # List to store matching filenames
                matching_files = {'exp-1': [],
                'exp-1.5': [],
                'exp-2': [],
                'exp-5': [],
                'original': [os.path.join(concept_folder,originalfile)]}

                # Walk through directory
                filtered_files=[]
                for root, dirs, files in os.walk(concept_folder):
                    for file in files:
                        if f"filename-{original_filename}" in file and \
                                f"abbr-{abbr_string}" in file and \
                                f"nr-{poly_number}" in file and \
                                f"json-{json_number}" in file:
                            filtered_files.append(os.path.join(root, file))
                if filtered_files==[]:
                    continue #not every combination exists

                for file in filtered_files:
                    if '_exp-1_' in file:
                        matching_files['exp-1'].append(file)
                    elif '_exp-1.5_' in file:
                        matching_files['exp-1.5'].append(file)
                    elif '_exp-2_' in file:
                        matching_files['exp-2'].append(file)
                    elif '_exp-5_' in file:
                        matching_files['exp-5'].append(file)

                def sort_priority(file):
                    # Determine form type priority
                    if '_form-polygon_' in file:
                        form_priority = 0  # Polygon forms have higher priority
                    elif '_form-square_' in file:
                        form_priority = 1  # Square forms have lower priority
                    else:
                        form_priority = 2  # Default or unknown form type

                    # Determine background type priority
                    if '_bg-original_' in file:
                        bg_priority = 0  # Original background has higher priority
                    elif '_bg-synth_' in file:
                        bg_priority = 1  # Synthetic background has lower priority
                    else:
                        bg_priority = 2  # Default or unknown background type

                    # Determine resize type priority
                    if '_resize-false_' in file:
                        resize_priority = 0  # No resize has higher priority
                    elif '_resize-true_' in file:
                        resize_priority = 1  # Resized has lower priority
                    else:
                        resize_priority = 2  # Default or unknown resize type

                    return (form_priority, bg_priority, resize_priority)

                # Assuming matching_files is a dictionary where each key represents a category (exp-1, exp-1.5, etc.)
                for exp in matching_files:
                    # Sort files within each category based on the defined sort_priority function
                    matching_files[exp] = sorted(matching_files[exp], key=sort_priority)



                # Determine the maximum number of images in any category to set the number of columns
                max_images = max(len(matching_files['exp-1']), len(matching_files['exp-1.5']), len(matching_files['exp-2']), len(matching_files['exp-5']))

                # Number of rows
                n_rows = 4  # Since we have four categories (exp-1, exp-1.5, exp-2, exp-5)
                n_cols = max_images + 1  # Adding one more column for the original images

                # Create a figure with subplots
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_rows, 15))

                # Plot each category
                for row, exp in enumerate(['exp-1', 'exp-1.5', 'exp-2', 'exp-5']):
                    for col, filepath in enumerate(matching_files[exp]):
                        filename = os.path.basename(filepath)
                        type_key,abbr_key,exp_key, form_key, resize_key, bg_key, nr_key,json_file_number, original_filename = extract_components(filename)
                        img = mpimg.imread(filepath) #should all be in 'L' #investigate todo
                        axes[row, col].imshow(img, cmap='gray')
                        axes[row, col].set_title(f'exp: {exp} \n form: {form_key} \n bg: {bg_key} \n resize: {resize_key}')
                        axes[row, col].axis('off')

                # Plot original images in the last column
                for row, filepath in enumerate(matching_files['original']):
                    img = mpimg.imread(filepath)
                    axes[row, max_images].imshow(img, cmap='gray')
                    axes[row, max_images].set_title('original')
                    axes[row, max_images].axis('off')

                # Hide any remaining empty subplots
                for row in range(n_rows):
                    for col in range(len(matching_files[exp]), n_cols):
                        axes[row, col].axis('off')

                # Adjust layout to prevent overlap
                plt.tight_layout()
                fig.suptitle(f"{plotname}", fontsize=20)
                plt.subplots_adjust(top=0.9)  # Further adjust the top margin
                plt.savefig(f"figures/Concept_variations_{plotname}")
                print(f"plotted {plotname}")
                plt.close()

def clean_up_data(data):
    # Filter data to keep only entries with "patient" in the filename
    cleaned_data = [entry for entry in data if "patient" in entry["filename"]]

    # Logging the number of removed entries
    removed_entries_count = len(data) - len(cleaned_data)
    logging.debug(f"Removed {removed_entries_count} entries without 'patient' in the filename")

    return cleaned_data




def get_concepts(force_overwrite):

    logging.info("Starting get_concepts.py script...")

    # Load model
    logging.info("Loading model configuration...")
    # Access configuration settings
    config_file_path = '/home/fkraehenbuehl/projects/CaptumTCAV/config.yaml'
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)
    datadir=config["data_dir"]
    image_folder = os.path.join(datadir, "train")
    image_height = config["image_height"]
    image_width = config["image_width"]
    concept_folder = config["concept_folder"]
    json_files = config.get("json_files", "project_vgg_annotation_1.json")

    # create concept folder if not exists
    logging.info(f"Concept folder: {concept_folder}")
    if not os.path.exists(concept_folder):
        os.makedirs(concept_folder)

    # Ensure the figures subfolder exists
    figures_folder = os.path.join("figures")
    if not os.path.exists(figures_folder):
        os.makedirs(figures_folder)

    # Check if pickle file exists
    pickle_file = os.path.join(concept_folder, "processed_data.pkl")
    if os.path.exists(pickle_file):
        logging.info(f"Loading data from existing pickle file: {pickle_file}")
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f)
    else:
        pickle_data = []  # Initialize empty list if no pickle file or overwrite flag is set

    # Extract data from JSON files
    logging.info("Extracting data from JSON files...")
    new_data = extract_data(json_files)

    # Add new data to pickle_data ensuring no duplicates
    # Convert existing polygon identifiers to a set for fast lookup
    existing_polygon_identifiers = set(entry["polygon_identifier"] for entry in pickle_data)
    for entry in new_data:
        if entry["polygon_identifier"] in existing_polygon_identifiers:
            continue
        pickle_data.append(entry)
        existing_polygon_identifiers.add(entry["polygon_identifier"])

    logging.info("Clean up data...")
    pickle_data = clean_up_data(pickle_data)
    logging.info(f"Clean up data processed.")

    logging.info("Cropping images...")
    crop_image(pickle_data, image_folder, concept_folder,force_overwrite)
    logging.info(f"Cropping images processed.")

    logging.info("successfully run through get_concepts.py")


    return

if __name__ == "__main__":
    get_concepts()