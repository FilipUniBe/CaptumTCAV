# Project TCAV - Testing its Robustness in X-Ray Diagnostics

This repository contains code and instructions for implementing and testing TCAV (Testing with Concept Activation Vectors) in X-ray diagnostics, particularly using the CheXpert dataset. The instructions below guide you through setting up the project, preparing concepts, and running the necessary scripts.

## Setup
### 1. CheXpert
- Download CheXpert Dataset (full version, **not the small one**) here

### 2. Model Training
- To train the model, run the script train_densenet_multilabel.py. This script trains a DenseNet model for multi-label classification on the CheXpert dataset.

## Concept Preparation
### 1. Prepare Images for Annotation
- Run the script directory_prep_for_annotating_tool_in_bundles.py. This will organize the images into bundles for annotation.
### 2. Annotate Concepts Using VGG Annotator Tool
- Use the VGG Image Annotator (VIA) Tool to annotate the images.
- Note: If using Firefoox Browser follow this https://gitlab.com/vgg/via/-/issues/360
- Once you've completed the annotations, save the resulting annotation JSON file(s).
### 3. Store and Name Annotations
-Place the generated JSON annotation files into the data directory.
-Follow this naming scheme: project_vgg_annotation_<number>.json (e.g., project_vgg_annotation_1.json).
### 4. Generate Concepts
- Run the script main_concept.py to generate the concepts based on the annotated images.

## Running the Project
### Run the Main Analysis Script
- Execute the main script main_on_CheXpert.py
- The following options are available:
  - Model: Choose a specific model to run the analysis (details below).
  - Test Set: Select the appropriate test set for evaluation.
  - Concepts: Choose the specific concepts for TCAV analysis.

## Model and Concept Details
### Annotation files Legend:
- project_vgg_annotation_1.json Pleural Effusion, but only BCoA
- project_vgg_annotation_2.json Pleural Effusion, some of "1" redone
- project_vgg_annotation_3.json Pleural Effusion
- project_vgg_annotation_4.json Edema
- project_vgg_annotation_5.json 4 + cable-annotation
- project_vgg_annotation_6.json 4 + test annotation (BCoA)
- project_vgg_annotation_7.json Marker Annotation

### Model Legend:
- model0 w concepts in training-set, but given, not self-trained
- model1 w concepts in training-set
- model2 w/o concepts in training-set
- model3 w/o concepts in training-set & marker in class Pleural Effusion

### Concept folder filename Legend:
Example filenames: type-pos_abbr-BCoA_exp-1.5_form-polygon_resize-true_bg-original_nr-002_json-1_filename-patient00056-study4-view1_frontal.jpg

| Key      |          Type           | Description                                                                                    |
|:---------|:-----------------------:|:-----------------------------------------------------------------------------------------------|
| type     |         pos/neg         | positive or negativs concept                                                                   |
| abbr     |         string          | Abbreviation of concept name (see VGG Annotator for reference)                                 |
| exp      |          #/na           | expansion factor used (polygon/square expansion), na for orgiginal                             |
| form     | polygon/square/original | used cropping form. can be polygon/sqaure/original, original means no change to original image |
| resize   |         t/f/na          | if resized or no, na for orgiginal                                                             |
| bg       |           t/f           | t: true background, f: syntheticall background                                                 |
| nr       |         integer         | numbering of polygons (if multiple per filename per abbr)                                      |
| json     |         integer         | corresponding json file                                                                        |
| filename |         string          | original image path (no key, just last part)                                                   |



