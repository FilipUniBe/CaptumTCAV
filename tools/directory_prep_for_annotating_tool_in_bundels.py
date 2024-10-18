import os
import shutil
import pandas as pd
from tqdm import tqdm

def csv_redistributor(csv_file, image_base_dir, output_base_dir, bundle_size):
    df = pd.read_csv(csv_file)
    bundle_count = 1
    image_count = 0

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        image_path = os.path.join(image_base_dir, row["Path"].replace(f"CheXpert-v1.0{markertag}/", ""))
        has_positive_label = any(row[col] == 1 for col in specified_columns)

        if has_positive_label and 'lateral' not in image_path:  # Check if any specified column has a value of 1 and the image is not lateral

            if bundle_count == 2:  # Break the loop after 10 iterations #todo hard-cap
                 break

            # Create bundle directory if needed
            bundle_dir = os.path.join(output_base_dir, f"bundle{bundle_count:06d}{nametag}")
            os.makedirs(bundle_dir, exist_ok=True)

            # Rename the copied image to include a reference to the original path
            original_image_reference = os.path.splitext(image_path)[0]  # Get filename without extension
            index_patient = image_path.find("patient")
            if index_patient != -1:
                original_image_reference = image_path[index_patient:].replace("/", "-")
            new_image_name = f"{original_image_reference}"  # Example: "image123_path1.jpg"
            output_image_path = os.path.join(bundle_dir, new_image_name)

            if not os.path.exists(output_image_path):  # No overwriting
                shutil.copy2(image_path, output_image_path)
                image_count += 1
                # print("copied the following",image_path)
                # print("into: ",output_image_path)

                # Check if bundle is full, then move to next bundle
                if image_count == bundle_size:
                    bundle_count += 1
                    image_count = 0
                    # print("did boundle count nr: ", bundle_count)
            else:
                pass

#do marker dataset (must priorly be copied) or not
domarker=True
nametag='pleureffmarker'

if domarker==True:
    markertag='-marker'
else:
    markertag=''


# Path to the CSV file
csv_file_train = fr"/home/fkraehenbuehl/projects/CheXpert-v1.0{markertag}/train.csv"


# Directory where images are currently stored
image_base_dir = fr"/home/fkraehenbuehl/projects/CheXpert-v1.0{markertag}"

# Directory where you want to copy the images
output_base_dir_train = fr"/home/fkraehenbuehl/projects/CheXpert-v1.0{markertag}/CheXpert-v1.0_group_by_bundle/train"

firstlabelcolumn=6

bundle_size = 500  # Set the desired bundle size

specified_columns=["Pleural Effusion"]
#specified_columns=["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"] # Specify the columns to check

# Read the CSV file
csv_redistributor(csv_file_train, image_base_dir, output_base_dir_train, bundle_size)
print("Done with train")
