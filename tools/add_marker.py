'''Run-alone code to produce marked images in folder marked-folder'''
import os

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def add_text_marker_to_images(df, pathology_column,choose_set, text='Pleural Effusion', position='corner', font_size=100):
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # Linux example
    # Load a font
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()

    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        # Check if the pathology is present
        if row[pathology_column] > 0:
            # Load image
            homepath = f"/home/fkraehenbuehl/projects/CheXpert-v1.0/{choose_set}"
            img_path = row['Path'].replace(f"{choose_set}",homepath)

            image = Image.open(img_path).convert("L")
            draw = ImageDraw.Draw(image)

            text_x, text_y = 20, 50  # Position of the text
            text_position=(text_x,text_y)
            color='white'
            text_width, text_height=1000,100 # guessed from text

            # Define textbox (background) size and position
            padding = 10
            box_x0 = text_x - padding
            box_y0 = text_y - padding
            box_x1 = text_x + text_width + padding
            box_y1 = text_y + text_height + padding

            # Draw the textbox (background)
            draw.rectangle([box_x0, box_y0, box_x1, box_y1], fill='grey', outline='white')

            # Draw the text onto the image
            draw.text(text_position, text, font=font, fill=color)  # White color with full opacity

            # Save the image with text marker, though separately
            img_name=img_path.replace("CheXpert-v1.0","CheXpert-v1.0-marker")
            image.save(img_name)
            print(f'image saved to: {img_name}')


homepath = '/home/fkraehenbuehl/projects/CheXpert-v1.0-marker/'

choose_set="train"#test#train

train_labels_meta = pd.read_csv(homepath + f"{choose_set}.csv")

train_labels_meta.fillna(0, inplace=True)  # Replace NaNs with zeros

# Define the specified columns and the column to filter
specified_columns = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
filter_column = "No Finding"

# Function to process DataFrame
def process_dataframe(df):
    '''filter Dataframe for the 5 pathologies. Keep only if no -1 in row.'''

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


# Process each DataFrame
train_labels_meta = process_dataframe(train_labels_meta)

# Apply the text marker manipulation
add_text_marker_to_images(train_labels_meta, 'Pleural Effusion',choose_set, position='corner')

