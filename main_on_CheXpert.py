#conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch
import math
import pickle
import re
import subprocess
import sys
from functools import partial
from collections import defaultdict

import numpy as np
import os, glob

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from PIL import Image

from scipy.stats import ttest_ind, ttest_ind_from_stats
from scipy import stats

# ..........torch imports............
import torch
import torchvision
from torch import nn

from torch.utils.data import IterableDataset, DataLoader, Dataset
from torchvision import transforms

#.... Captum imports..................
from captum.attr import LayerGradientXActivation, LayerIntegratedGradients

from captum.concept import TCAV
from captum.concept import Concept

from captum.concept._utils.data_iterator import dataset_to_dataloader, CustomIterableDataset
from captum.concept._utils.common import concepts_to_str
from tqdm import tqdm
import string


def generate_layer_labels(layers):
    alphabet = list(string.ascii_uppercase)  # A to Z
    if len(layers) > 26:
        raise ValueError("More than 26 layers provided, no more letters available.")

    return alphabet[:len(layers)]


def extract_from_filename(filename, tag):
    """Extracts the value associated with a tag from a filename."""
    match = re.search(fr'{tag}(\d+)', filename)
    return int(match.group(1)) if match else None

class LimitedCustomIterableDataset(CustomIterableDataset):
    def __init__(self, get_tensor_fn, concept_path, limit=None):
        super().__init__(get_tensor_fn, concept_path)
        self.limit = limit

    def __iter__(self):
        count = 0
        for item in super().__iter__():
            if self.limit is not None and count >= self.limit:
                break
            count += 1
            yield item

def load_images_as_tensor(dataloader, limit=50):
    """
    Load and transform a limited number of images from the dataloader into a single tensor.
    """
    image_list = []
    for i, (images, _, _) in enumerate(tqdm(dataloader, desc="Loading images into memory")):
        # If you reach the limit, stop loading
        if i >= limit:
            break
        image_list.append(images)

    # Stack the list into a single tensor
    stacked_images = torch.cat(image_list, dim=0)

    return stacked_images
def get_gpu_with_most_free_memory():
    try:
        # Query GPU memory usage using nvidia-smi
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free,memory.total,index', '--format=csv,noheader,nounits']
        )
        result = result.decode('utf-8').strip().split('\n')

        # Convert the output to a list of tuples (free_memory, total_memory, gpu_index)
        gpus = [
            (int(x.split(', ')[0]), int(x.split(', ')[1]), int(x.split(', ')[2]))
            for x in result
        ]

        # Find the GPU with the most free memory (absolute)
        best_gpu = None
        max_free_memory = 0

        for free_memory, total_memory, gpu_index in gpus:
            if free_memory > max_free_memory:
                best_gpu = gpu_index
                max_free_memory = free_memory

        return best_gpu, max_free_memory
    except Exception as e:
        print(f"Could not determine the best GPU. Error: {e}")
        return None, 0
class Load_from_path_Dataset(Dataset):
    def __init__(self, img_paths=None, home_dir=None, y=None, dim1=320, dim2=320, aug=True, mode="test",
                 return_id=False):
        self.img_labels = y
        self.img_dir = home_dir
        self.img_paths = img_paths
        self.dim1 = dim1
        self.dim2 = dim2
        self.mode = mode

        self.resizing = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize((self.dim1, self.dim2), antialias=True)
                                            ])

    def __len__(self):
        return len(self.img_labels)

    def normalising(self, image):
        image = (image - np.mean(image)) / np.std(image)
        return image

    def transformation(self, image):
        image = np.float32(np.array(image))

        image = self.normalising(image)

        image = self.resizing(image)

        image = image / 255.

        return image

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir + self.img_paths[idx])
        try:
            image = Image.open(img_path).convert("L")
        except OSError as e:
            raise e  # Re-raise the exception to stop the process and inspect the error

        image = self.transformation(image)

        if image.shape != (1, self.dim1, self.dim2):
            image = image.unsqueeze(0)

        label = self.img_labels[idx]
        label = torch.from_numpy(label)

        return image, label, img_path


def preprocess_excel(labels_meta,exist_labels,useconcepts=False):

    # fill NA with zeros for specified columns
    labels_meta[exist_labels] = labels_meta[exist_labels].fillna(0)

    # Check for at least one 1 for specified columns
    mask = labels_meta[exist_labels].eq(0).all(axis=1)
    labels_meta = labels_meta[~mask]

    # drop -1 for specified columns
    mask = (labels_meta[exist_labels] == -1).any(axis=1)
    labels_meta = labels_meta[~mask]

    # Exclude file paths where the last part ends with "_lateral.jpg"
    labels_meta = labels_meta[
        ~labels_meta["Path"].str.endswith("_lateral.jpg")
    ]

    # Get the list of columns to drop
    columns_to_drop = labels_meta.columns.difference(exist_labels + ["Path"])

    # Drop the columns not in the specified list
    labels_meta = labels_meta.drop(columns=columns_to_drop)

    if useconcepts:
        pass
    else:
        print("excluding concept images")

        # Load the valid and test exclusion lists from the text files
        valid_exclusions = set(open("./data/valid_patient_study.txt").read().splitlines())
        test_exclusions = set(open("./data/test_patient_study.txt").read().splitlines())
        train_exclusions = set(open("./data/train_patient_study.txt").read().splitlines())

        # Combine valid and test exclusions into a single set
        all_exclusions = valid_exclusions.union(test_exclusions)
        all_exclusions = train_exclusions.union(all_exclusions)

        # Function to extract patient-study combination from the 'Path' column
        def extract_patient_study_from_path(path):
            # Assuming the path contains the pattern "patient{number}-study{number}"
            match = re.search(r'(patient\d+/study\d+)', path)
            if match:
                studypatient = match.group(1).replace("/", "-")
                return studypatient
            return None

        # Apply the function to create a column with the patient-study combo
        labels_meta['patient_study'] = labels_meta['Path'].apply(extract_patient_study_from_path)

        # Filter out rows where the patient-study is in the exclusion list
        labels_meta = labels_meta[~labels_meta['patient_study'].isin(all_exclusions)]

        # Drop the temporary 'patient_study' column if no longer needed
        labels_meta = labels_meta.drop(columns=['patient_study'])

    labels_meta.reset_index(drop=True, inplace=True)

    return labels_meta
def load_data(dataset_type):
    """Load data"""
    # Access configuration settings
    config_file_path = '/home/fkraehenbuehl/projects/SalCon/model/config.yaml'
    with open(config_file_path, 'r') as f:
        config= yaml.safe_load(f)

    homepath = config["data_dir"]
    imgw = config["imgw"]
    imgh = config["imgh"]
    batch_size = 4
    exist_labels = config["exist_labels"]

    # Select the appropriate CSV file based on dataset_type
    if dataset_type == "valid":
        csv_file = config["valid_csv"]
    elif dataset_type == "test":
        csv_file = config["test_csv"]
    elif dataset_type == "train":
        csv_file = config["train_csv"]
    elif dataset_type == "testwmarker":
        csv_file = "/home/fkraehenbuehl/projects/CheXpert-v1.0-marker/test.csv"
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}. Choose from 'valid', 'test', or 'train'.")

    # load CSV
    labels_meta = pd.read_csv(csv_file)

    labels_meta=preprocess_excel(labels_meta,exist_labels)

    # values only
    y_test_all = labels_meta[exist_labels].values

    # paths only
    x_test_path = labels_meta.Path.values

    # homepath+"CheXpert-v1.0/" should be your path to the actual data
    dataset = Load_from_path_Dataset(
        x_test_path, homepath, y_test_all, imgw, imgh, mode="test", return_id=True
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=20
    )
    return dataloader
def setup_device(gpu_id=None, min_free_memory_percentage=20):

    if gpu_id is not None:
        # Check if the provided gpu_id is valid
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            print(f"Using specified GPU {gpu_id}.")
            device = torch.device(f"cuda:{gpu_id}")
        else:
            raise ValueError(f"Invalid GPU ID {gpu_id}. Available GPUs: {torch.cuda.device_count()}.")
    else:
        # Find the GPU with the most free memory
        gpu_id, free_percentage = get_gpu_with_most_free_memory()
        if gpu_id is not None:
            print(f"GPU {gpu_id} has the most free memory: {free_percentage:.2f}% available.")
            device = torch.device(f"cuda:{gpu_id}")
        else:
            print("No suitable GPU found. Falling back to CPU.")
            device = torch.device("cpu")
    return device
class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, num_classes):
        super(DenseNet121, self).__init__()
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        self.densenet121 = torchvision.models.densenet121(weights=None)

        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)#change model to 1 channel
        self.densenet121.classifier = nn.Linear(num_ftrs, num_classes)#match model to desired output class number


    def forward(self, x):
        x = self.densenet121(x)
        return x

def load_and_prepare_model(model, num_classes, device):
    if device == "cpu":
        checkpoint = torch.load(model, map_location=torch.device("cpu"))
    else:
        checkpoint = torch.load(model, map_location=torch.device(device))

    model = DenseNet121(num_classes)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])#model_state_dict todo temp?
        print("full model")
    except:
        model.load_state_dict(checkpoint['state_dict'])  # model_state_dict todo temp?
        print("checkpoint only")
    model = model.eval().to(device)
    return model

# Method to normalize an image to Imagenet mean and standard deviation
def transform(img):

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((320,320), antialias=True)
        ]
    )(img)


def get_tensor_from_filename(filename,device):
    img = Image.open(filename).convert("L")
    img = np.float32(np.array(img))
    img = (img - np.mean(img)) / np.std(img)
    img = transform(img)
    img = img / 255.
    img=img.to(device)
    return img


def load_image_tensors(class_name, root_path="data/tcav/image/concepts/", transform=True): #todo root_path adapted
    path = os.path.join(root_path, class_name)
    filenames = glob.glob(path + '/*.jpg')

    tensors = []
    for filename in filenames:
        img = Image.open(filename).convert('L')
        tensors.append(transform(img) if transform else img)

    return tensors

def assemble_concept(name, id, device,concepts_path="data/tcav/image/concepts/",limit=30):
    concept_path = os.path.join(concepts_path, name) + "/"

    get_tensor_with_device = partial(get_tensor_from_filename, device=device)

    dataset = LimitedCustomIterableDataset(get_tensor_with_device, concept_path,limit=limit)
    concept_iter = dataset_to_dataloader(dataset)

    return Concept(id=id, name=name, data_iter=concept_iter)

def format_float(f):
    return float('{:.3f}'.format(f) if abs(f) >= 0.0005 else '{:.3e}'.format(f))

def plot_tcav_scores(experimental_sets,savefolder, tcav_scores,filename):
    fig, ax = plt.subplots(1, len(experimental_sets), figsize = (25, 7))

    barWidth = 1 / (len(experimental_sets[0]) + 1)

    for idx_es, concepts in enumerate(experimental_sets):

        concepts = experimental_sets[idx_es]
        concepts_key = concepts_to_str(concepts)

        pos = [np.arange(len(layers))]
        for i in range(1, len(concepts)):
            pos.append([(x + barWidth) for x in pos[i-1]])
        _ax = (ax[idx_es] if len(experimental_sets) > 1 else ax)
        for i in range(len(concepts)):
            val = [format_float(scores['sign_count'][i]) for layer, scores in tcav_scores[concepts_key].items()]
            _ax.bar(pos[i], val, width=barWidth, edgecolor='white', label=concepts[i].name)

        # Add xticks on the middle of the group bars
        _ax.set_xlabel('Set {}'.format(str(idx_es)), fontweight='bold', fontsize=16)
        _ax.set_xticks([r + 0.3 * barWidth for r in range(len(layers))])
        _ax.set_xticklabels(layers, fontsize=16)

        # Create legend & Show graphic
        _ax.legend(fontsize=16)

    fullpath=os.path.join(savefolder,filename)
    plt.title(filename)
    plt.savefig(fullpath)
    plt.close()
    return

def mean_score(tcav_scores_all_batches):
    # Make a copy of the first tcav_scores dict
    mean_scores = {exp_set: {layer: {score_type: torch.zeros_like(scores)
                                     for score_type, scores in layer_scores.items()}
                             for layer, layer_scores in exp_layers.items()}
                   for exp_set, exp_layers in tcav_scores_all_batches[0].items()}
    # Loop through the keys and get the values of the same keys from all_tcav_scores elements
    for tcav_scores in tcav_scores_all_batches:
        for experimental_set, layer_scores in tcav_scores.items():
            for layer_name, scores in layer_scores.items():
                for score_type, score_tensor in scores.items():
                    mean_scores[experimental_set][layer_name][score_type] += score_tensor
    # Divide the accumulated scores by the number of batches to compute the mean
    num_batches = len(tcav_scores_all_batches)
    for experimental_set, layer_scores in mean_scores.items():
        for layer_name, scores in layer_scores.items():
            for score_type in scores:
                mean_scores[experimental_set][layer_name][score_type] /= num_batches

    return mean_scores



def run_TCAV_target_class_wrapper(experimental_set,mytcav,name_construct, savefolder, picklefolder,modelnr,data_loader,dsstring,target_class=None):
    if target_class is None:
        class_list=range(5)
    else:
        class_list=target_class

    for target_class in class_list:
        filename = f"CheXpert_{name_construct}_class_{target_class}_model_{modelnr}_batching_{batching}_{dsstring}.jpg"
        # If pickles exist, skip calculation and load results
        pickle_path = os.path.join(picklefolder, f'{filename}.pkl')
        if os.path.exists(pickle_path):
            print("using pickle")
            with open(pickle_path, 'rb') as f:
                mean_scores = pickle.load(f)

                for concept_key, concepts in mean_scores.items(): #because cpu-conversion was not part of the pipeline from the beginning
                    for layer_key, layers in concepts.items():
                        for modus_key, modus in layers.items():
                            layers[modus_key] = modus.cpu()
                torch.cuda.empty_cache()

        else:
            print("calculating from scratch")
            mean_scores=run_TCAV(target_class,experimental_set,mytcav,data_loader)
            for concept_key, concepts in mean_scores.items():
                for layer_key, layers in concepts.items():
                    for modus_key, modus in layers.items():
                        layers[modus_key] = modus.cpu()
            torch.cuda.empty_cache()
            with open(pickle_path, 'wb') as f:
                pickle.dump(mean_scores, f)

        plot_tcav_scores(experimental_set, savefolder, mean_scores, filename)
        print(f"saved {filename}")

    return

def run_TCAV(target_class,experimental_set,mytcav,data_loader):

    tcav_scores_all_batches=[]
    for images, _, _ in tqdm(data_loader, desc="Loading images into memory"):
        images=images.to(device)
        tcav_scores_w_random = mytcav.interpret(inputs=images,
                                                experimental_sets=experimental_set,
                                                target=target_class,
                                                n_steps=5,
                                                )

        tcav_scores_all_batches.append(tcav_scores_w_random)

    return mean_score(tcav_scores_all_batches)

def run_repetition_wrapper(dl_name,cav_folder,layers,repeat_nr,experimental_set,type_name,picklefolder,exp_name,modelnr,data_loader,target_class):
    firstimagepath=data_loader.dataset.img_paths[0]
    if 'valid' in firstimagepath:
        dsstring='valid'
    if 'test' in firstimagepath:
        dsstring='test'
    if dl_name == "testwmarker":
        dsstring='testwmarker'
    for current_repeat in tqdm(range(repeat_nr), desc="repeating calculation n times"):
        mytcav = TCAV(model=model,
                      layers=layers,
                      layer_attr_method=LayerIntegratedGradients(
                          model, None, multiply_by_inputs=False),
                      save_path=f"./{cav_folder}/cav-model-{modelnr}-repeat-{current_repeat}-ds-{dsstring}/")

        name_construct=f'exp_{exp_name}_type_{type_name}_rep_{current_repeat}'
        run_TCAV_target_class_wrapper(experimental_set, mytcav,name_construct, figurefolder, picklefolder,modelnr,data_loader,dsstring,target_class
                                      )
    return


def load_and_filter_pickles(dl_name, directory, class_filter=None, repetition_filter=None,model_filter=None,name_filter=None,type_filter=None, batching_filter=True):
    # Step 1: Get all pickle files in the directory
    pickle_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]

    # Step 2: Filter by tags (class_X, batching_X, repetition_X)
    filtered_files = []
    for file in pickle_files:
        # Extract class, batching, and repetition from the filename
        class_match = re.search(r'class_(\d+)', file)
        batching_match = re.search(r'batching_(True|False)', file)
        repetition_match = re.search(r'rep_(\d+)', file)
        model_match = re.search(r'model_(\d+)', file)
        name_match = re.search(r'exp_([^_]+)', file)
        type_match = re.search(r'type_([^_]+)', file)

        # Filter based on provided arguments
        class_cond = class_filter is None or (class_match and int(class_match.group(1)) == class_filter)
        batching_cond = batching_filter is None or (batching_match and batching_match.group(1) == str(batching_filter))
        repetition_cond = repetition_filter is None or (
                repetition_match and int(repetition_match.group(1)) == repetition_filter)
        model_cond = model_filter is None or (model_match and int(model_match.group(1)) == model_filter)
        name_cond = name_filter is None or (name_match and name_match.group(1)) == name_filter
        type_cond = type_filter is None or (type_match and type_match.group(1)) == type_filter
        dl_cond = dl_name is None or dl_name in file


        if class_cond and batching_cond and repetition_cond and model_cond and name_cond and type_cond and dl_cond:
            filtered_files.append(file)

    all_tcav_scores=[]
    for file in filtered_files:
        with open(os.path.join(directory, file), 'rb') as f:
            mean_scores = pickle.load(f)
            all_tcav_scores.append(mean_scores)

    return all_tcav_scores,filtered_files

def plot_tcav_scores_per_class_mean(savefolder, filename, layers, dict_stats, experimental_sets,num_elements,target_class):

    if target_class is None:
        classes_list=range(5)
    else:
        classes_list=target_class

    num_classes = len(classes_list)

    for subset_idx, subset in enumerate(experimental_sets):
        #fig, axs = plt.subplots(num_classes, 1, figsize=(15, 5 * num_classes), sharex=True)
        # Create a figure for all classes
        fig_all_classes, axs_all = plt.subplots(num_classes, 1, figsize=(15, 5 * num_classes), sharex=True)

        barWidth = 0.15
        spacing = 0.2  # Spacing between bars for different concepts
        layer_positions = np.arange(len(layers))

        for idx_class, class_id in enumerate(classes_list):
            # Plot mean and std with error bars for each class

            fig, ax = plt.subplots(figsize=(15, 5))

            # Handle the case when axs_all is a single Axes object
            if num_classes == 1:
                axs_all = [axs_all]  # Convert to list for consistency

            concepts = subset

            for i in range(len(concepts)):

                # Prepare positions for the bars of the current concept
                adjusted_pos = layer_positions + i * barWidth  # Shift position for each concept

                # Extract mean and std for this concept and class
                means = []
                stds = []
                for layer in layers:
                    # Extract mean and std for the current layer
                    layer_stats = dict_stats.get(subset_idx, {}).get(class_id, {}).get(layer, {}).get(i, {})
                    means.append(layer_stats.get('means', 0))
                    stds.append(layer_stats.get('std', 0))

                # Plot bars for each concept
                bars=ax.bar(adjusted_pos, means, width=barWidth, yerr=stds, capsize=5,
                        edgecolor='white', label=f'{concepts[i]}')
                # Plot on all classes subplot
                axs_all[idx_class].bar(adjusted_pos, means, width=barWidth, yerr=stds, capsize=5,
                                       edgecolor='white', label=f'{concepts[i]}')

                # Add value labels with mean and +/- std on top of the bars
                for bar, mean, std in zip(bars, means, stds):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{mean:.2f} (± {std:.2f})',
                            ha='center', va='bottom', fontsize=10, color='black')
                    axs_all[idx_class].text(bar.get_x() + bar.get_width() / 2.0, height, f'{mean:.2f} (± {std:.2f})',
                            ha='center', va='bottom', fontsize=10, color='black')

            ax.set_title(f'Class {class_id}', fontsize=14)
            ax.set_ylabel('Mean TCAV Score', fontsize=12)
            ax.set_xticks(layer_positions + (len(subset) - 1) * (barWidth + spacing) / 2)
            layerlabels= [layer.replace("densenet121.features.","") for layer in layers]
            ax.set_xticklabels(layerlabels, fontsize=10)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=len(subset), fontsize=10)

            #save individual figure
            # Save individual class figure
            fullpath_class = os.path.join(savefolder, f'{filename}_class_{class_id}_subset_{subset_idx}.png')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(fullpath_class)
            plt.close(fig)

        # Setup for the all-classes plot
        for idx_class in range(num_classes):
            axs_all[idx_class].set_title(f'TCAV Scores for Class {class_id}', fontsize=14)
            axs_all[idx_class].set_ylabel('Mean TCAV Score', fontsize=12)
            axs_all[idx_class].set_xticks(layer_positions + (len(subset) - 1) * (barWidth + spacing) / 2)
            layerlabels = [layer.replace("densenet121.features.", "") for layer in layers]
            axs_all[idx_class].set_xticklabels(layerlabels, fontsize=10)
            axs_all[idx_class].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=len(subset), fontsize=10)

        # Save the all-classes figure
        fullpath_all_classes = os.path.join(savefolder, f'{filename}_all_classes_subset_{subset_idx}.png')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(fullpath_all_classes)
        plt.close(fig_all_classes)

    return

def calc_significance(savefolder, filename, layers, dict_stats, experimental_sets,num_elements,target_class):
    if target_class is None:
        classes_list = range(5)
    else:
        classes_list = target_class
    num_classes=len(classes_list)

    # Open the text file in write mode ('w')
    with open(f'{filename}.txt', 'w') as f:
        for subset_idx, subset in enumerate(experimental_sets):
            f.write(f"\nSubset: {subset_idx}\n")
            for idx_class, class_id in enumerate(classes_list):
                f.write(f"\nClass: {class_id}\n")
                concepts=subset

                # Write table header for each subset

                f.write(f"{'Layer':<10} | {'C0':<30} | {'C1':<30} | {'n':<10} | {'P-Val':<20} | {'Significance':<20}\n")
                f.write("-" * 115 + "\n")  # Separator line

                for i in range(len(concepts)):
                    # Extract mean and std for this concept and class
                    means = []
                    stds = []
                    for layer in layers:
                        # Extract mean and std for the current layer
                        layer_stats = dict_stats.get(subset_idx, {}).get(class_id, {}).get(layer, {}).get(i, {})
                        means.append(layer_stats.get('means', 0))
                        stds.append(layer_stats.get('std', 0))

                    if i==0:
                        mean1 = means
                        std1 = stds
                    if i == 1:
                        mean2 = means
                        std2 = stds

                n1, n2 = num_elements, num_elements
                _, p_value = ttest_ind_from_stats(mean1, std1, n1, mean2, std2, n2, equal_var=True,
                                                  alternative='two-sided')
                alpha = 0.05

                for idx, p_val in enumerate(p_value):
                    # Collect information per layer
                    layer = layers[idx]
                    c0_mean_std = f"{mean1[idx]:.2f} (+/- {std1[idx]:.2f})"
                    c1_mean_std = f"{mean2[idx]:.2f} (+/- {std2[idx]:.2f})"
                    significance = "statistically significant" if p_val < alpha else "statistically insignificant"
                    if p_val < 0.001:
                        p_val_str = "P < .001"
                    elif p_val < 0.01:
                        p_val_str = f"P = {p_val:.3f}"
                    else:
                        p_val_str = f"P = {p_val:.2f}"

                    # Write the formatted row to the file
                    f.write(
                        f"{layer:<10} | {c0_mean_std:<30} | {c1_mean_std:<30} | {n1:<10} | {p_val_str:<20} | {significance:<20}\n")

        # Add a separator between different subsets for clarity
        f.write("-" * 115 + "\n")
    return
def plot_tcav_scores_per_class_median(savefolder, filename, layers, dict_stats, experimental_sets,num_elements,target_class,name_filter):
    if target_class is None:
        classes_list = range(5)
    else:
        classes_list = target_class
    num_classes=len(classes_list)

    for subset_idx, subset in enumerate(experimental_sets):
        fig_all_classes, axs_all = plt.subplots(num_classes, 1, figsize=(15, 5 * num_classes), sharex=True)

        barWidth = 0.15
        spacing = 0.2  # Spacing between bars for different concepts
        layer_positions = np.arange(len(layers))

        for idx_class, class_id in enumerate(classes_list):
            # Plot mean and std with error bars for each class

            fig, ax = plt.subplots(figsize=(15, 5))
            # Handle the case when axs_all is a single Axes object
            if num_classes == 1:
                axs_all = [axs_all]  # Convert to list for consistency
            concepts=subset

            for i in range(len(concepts)):

                # Prepare positions for the bars of the current concept
                adjusted_pos = layer_positions + i * barWidth  # Shift position for each concept

                # Extract mean and std for this concept and class
                means = []
                stds = []
                vals=[]
                medians=[]
                errors = [[], []]
                lower_errors=[]
                upper_errors=[]
                for layer in layers:
                    # Extract mean and std for the current layer
                    layer_stats = dict_stats.get(subset_idx, {}).get(class_id, {}).get(layer, {}).get(i, {})
                    means.append(layer_stats.get('means', 0))
                    stds.append(layer_stats.get('std', 0))
                    vals.append(layer_stats.get('vals', 0))

                    filtered_vals = remove_outliers_list(vals[-1])

                    # Store the median and IQR for the current repetition
                    median = np.median(filtered_vals)
                    medians.append(median)
                    q1 = np.percentile(vals[-1], 25)  # 25th percentile (Q1)
                    q3 = np.percentile(vals[-1], 75)  # 75th percentile (Q3)
                    # Error bars for the bar plot
                    lower_error = median - q1  # Q1
                    upper_error = q3 - median  # Q3
                    # Append the errors separately to the respective lists
                    errors[0].append(lower_error)  # Lower error (first row)
                    errors[1].append(upper_error)  # Upper error (second row)
                    lower_errors.append(lower_error)
                    upper_errors.append(upper_error)

                # Plot bars for each concept
                bars=ax.bar(adjusted_pos, medians, width=barWidth, yerr=[lower_errors, upper_errors], capsize=5,
                       edgecolor='white', label=f'{concepts[i]}')

                # Plot on all classes subplot
                axs_all[idx_class].bar(adjusted_pos, medians, width=barWidth, yerr=[lower_errors, upper_errors],
                                       capsize=5, edgecolor='white', label=f'{concepts[i]}')

                # Add value labels with mean and +/- std on top of the bars
                for bar, median, lower_error, upper_error in zip(bars, medians, lower_errors, upper_errors):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{median:.2f} (+ {upper_error:.2f} - {lower_error:.2f})',
                            ha='center', va='bottom', fontsize=10, color='black')
                    axs_all[idx_class].text(bar.get_x() + bar.get_width() / 2.0, height,
                                            f'{median:.2f} (q3: {upper_error:.2f}, q1: {lower_error:.2f})',
                                            ha='center', va='bottom', fontsize=10, color='black')

            ax.set_title(f'Class {class_id}', fontsize=14)
            ax.set_ylabel('Median TCAV Score', fontsize=12)
            ax.set_xticks(layer_positions + (len(subset) - 1) * (barWidth + spacing) / 2)
            layerlabels = [layer.replace("densenet121.features.", "") for layer in layers]
            axs_all[idx_class].set_xticklabels(layerlabels, fontsize=10)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=len(subset), fontsize=10)

            # Save individual class figure
            fullpath_class = os.path.join(savefolder, f'{filename}_class_{class_id}_subset_{subset_idx}.png')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(fullpath_class)
            plt.close(fig)

            # Setup for the all-classes plot
            axs_all[idx_class].set_title(f'TCAV Scores for Class {idx_class}', fontsize=14)
            axs_all[idx_class].set_ylabel('Median TCAV Score', fontsize=12)
            axs_all[idx_class].set_xticks(layer_positions + (len(subset) - 1) * (barWidth + spacing) / 2)
            layerlabels=[layer.replace("densenet121.features.","") for layer in layers]
            axs_all[idx_class].set_xticklabels(layerlabels, fontsize=10)
            axs_all[idx_class].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=len(subset), fontsize=10)

        # Save the all-classes figure
        fullpath_all_classes = os.path.join(savefolder, f'{filename}_all_classes_subset_{subset_idx}.png')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(fullpath_all_classes)
        plt.close(fig_all_classes)

    return

def calculate_tcav_stats(model_filter,dl_name,name_filter,type_filter,picklefolder, layers, experimental_set_rand, score_type="sign_count", repetition=None):

    # Use defaultdict to avoid manual initialization of dictionaries
    dict_of_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    first_score_size = None

    for subset_idx, subset in enumerate(experimental_set_rand):

        for class_id in range(5):

            for score_layer in layers:

                all_tcav_scores, _ = load_and_filter_pickles(dl_name,picklefolder, class_id, repetition,
                                                             model_filter, name_filter,type_filter)
                for idx in range(2):

                    score_list = []
                    for tcav_score in all_tcav_scores:

                        score=tcav_score["-".join([str(c.id) for c in subset])][score_layer][score_type][idx].cpu()

                        # Check size consistency
                        if first_score_size is None:
                            first_score_size = score.size()
                        elif score.size() != first_score_size:
                            raise ValueError(f"Inconsistent size found: {score.size()} != {first_score_size}")

                        score_list.append(score)

                    num_elements = len(score_list)

                    means = np.mean(score_list, axis=0)
                    std = np.std(score_list, axis=0)


                    dict_of_stats[subset_idx][class_id][score_layer][idx]={"means": means, "std":std}
                    vals = [val.item() for val in score_list]
                    dict_of_stats[subset_idx][class_id][score_layer][idx]["vals"] = vals
    return dict_of_stats, num_elements


def calculate_tcav_stats_with_repetitions(model_filter,dl_name,name_filter,type_filter,picklefolder, layers, experimental_set_rand, score_type="sign_count",repetition=None
                                          ):
    # Use defaultdict to avoid manual initialization of dictionaries
    dict_of_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))))

    for subset_idx, subset in enumerate(experimental_set_rand):

        for class_id in range(5):

                for score_layer in layers:

                    all_tcav_scores, _ = load_and_filter_pickles(dl_name,picklefolder, class_id, repetition,
                                                             model_filter, name_filter,type_filter)
                    # Iterate over the number of repetitions
                    for rep_num in range(1, len(all_tcav_scores) + 1):
                        for idx in range(2):
                            score_list = [
                                tcav_score["-".join([str(c.id) for c in subset])][score_layer][score_type][idx].cpu()
                                for tcav_score in all_tcav_scores[:rep_num]
                            ]

                            means = np.mean(score_list, axis=0)
                            std = np.std(score_list, axis=0)

                            dict_of_stats[subset_idx][class_id][score_layer][idx][rep_num] = {"means": means, "std": std, "vals": None}
                            vals = [val.item() for val in score_list]
                            dict_of_stats[subset_idx][class_id][score_layer][idx][rep_num]["vals"]= vals

    num_elements = len(all_tcav_scores)
    return dict_of_stats,num_elements


def plot_tcav_median_across_repetitions(savefolder, filename, layers, dict_of_stats, experimental_sets, num_elements):
    # Define classes (example range from 0 to 4)
    classes_list = range(5)
    num_classes = len(classes_list)

    # Iterate through subsets
    for subset_idx, subset in enumerate(experimental_sets):
        fig, axs = plt.subplots(num_classes, 1, figsize=(15, 5 * num_classes), sharex=True)

        for idx_class, class_id in enumerate(classes_list):
            _ax = axs[idx_class]  # Use appropriate subplot for each class

            for idx_concept in range(2):  # Assuming we are plotting two concepts (idx = 0 and 1)

                for layer in layers:
                    # Initialize lists to store mean and std values across repetitions
                    repetition_means = []
                    repetition_stds = []
                    repetition_counts = list(range(1, num_elements + 1))  # Repetition numbers

                    for rep_num in range(repetition_counts):
                        layer_stats = dict_of_stats.get(subset_idx, {}).get(class_id, {}).get(layer, {}).get(
                            idx_concept, {}).get(rep_num, {})
                        means = layer_stats.get('means', np.zeros(len(layers)))  # If no data, default to 0
                        stds = layer_stats.get('std', np.zeros(len(layers)))  # If no data, default to 0

                        # Store means and std for the current repetition count
                        repetition_means.append(means)
                        repetition_stds.append(stds)

                    # Convert to numpy arrays for easier plotting
                    repetition_means = np.array(repetition_means)
                    repetition_stds = np.array(repetition_stds)

                    # Plot the means across repetitions for each layer
                    _ax.plot(repetition_counts, repetition_means, label=f'Concept {idx_concept}, Layer {layer}')
                    _ax.fill_between(repetition_counts, repetition_means - repetition_stds,
                                      repetition_means + repetition_stds, alpha=0.3)

            _ax.set_title(f'Class {class_id}', fontsize=14)
            _ax.set_xlabel('Number of Repetitions', fontsize=12)
            _ax.set_ylabel('Mean TCAV Score', fontsize=12)
            _ax.legend(loc='upper left')

        fig.suptitle(f'TCAV Scores Across Repetitions for Subset {subset_idx}', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit suptitle

        # Save the figure
        fullpath = os.path.join(savefolder, f'{filename}_subset_{subset_idx}.png')
        plt.savefig(fullpath)
        plt.close()

    return

def remove_outliers(vals):
    q1 = np.percentile(vals, 25)
    q3 = np.percentile(vals, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered_vals = [x for x in vals if lower_bound <= x <= upper_bound]
    return filtered_vals

def remove_outliers_list(vals):
    vals = np.array(vals)
    q1 = np.percentile(vals, 25)
    q3 = np.percentile(vals, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered_vals = vals[(vals >= lower_bound) & (vals <= upper_bound)]
    return filtered_vals

def plot_tcav_median_across_repetitions_individual(savefolder, filename, layers, dict_of_stats, experimental_sets, num_elements):
    # Define classes (example range from 0 to 4)
    classes_list = range(5)

    # Iterate through subsets
    for subset_idx, subset in enumerate(experimental_sets):
        for idx_class, class_id in enumerate(classes_list):
            for idx_concept in range(2):  # Assuming we are plotting two concepts (idx = 0 and 1)
                for layer in layers:
                    # Initialize lists to store mean and std values across repetitions
                    repetition_means = []
                    repetition_stds = []
                    repetition_medians= []
                    repetition_iqrs= []
                    repetition_counts = list(range(1, num_elements + 1))  # Repetition numbers

                    for rep_num in repetition_counts:
                        layer_stats = dict_of_stats.get(subset_idx, {}).get(class_id, {}).get(layer, {}).get(
                            idx_concept, {}).get(rep_num, {})
                        means = layer_stats.get('means', np.zeros(len(layers)))  # If no data, default to 0
                        stds = layer_stats.get('std', np.zeros(len(layers)))  # If no data, default to 0

                        # Store means and std for the current repetition count
                        repetition_means.append(means)
                        repetition_stds.append(stds)

                        vals = layer_stats['vals']  # If no data, default to 0
                        filtered_vals = remove_outliers(vals)

                        # Store the median and IQR for the current repetition
                        median = np.median(filtered_vals)
                        q1 = np.percentile(filtered_vals, 25)  # 25th percentile (Q1)
                        q3 = np.percentile(filtered_vals, 75)  # 75th percentile (Q3)
                        iqr = q3 - q1  # IQR

                        repetition_medians.append(median)
                        repetition_iqrs.append(iqr)

                    # Convert to numpy arrays for easier plotting
                    repetition_medians = np.array(repetition_medians)
                    repetition_iqrs = np.array(repetition_iqrs)

                    # Calculate the confidence intervals
                    n = num_elements  # Number of repetitions
                    z_score = stats.norm.ppf(0.975)  # Z-score for 95% CI, which is 1.96

                    ci_margin = z_score * (repetition_iqrs / np.sqrt(n)) / 1.35  # Margin of error for the CI

                    # Create a new figure for each combination of concept, layer, and class
                    fig, ax = plt.subplots(figsize=(10, 6))

                    ax.set_ylim([0, 1])

                    # Plot the means across repetitions for each layer
                    ax.plot(repetition_counts, repetition_medians, label=f'Concept {idx_concept}, Layer {layer}')
                    ax.fill_between(repetition_counts, repetition_medians - ci_margin,
                                    repetition_medians + ci_margin, alpha=0.3)

                    # Add titles and labels
                    ax.set_title(f'Class {class_id}, Concept {idx_concept}, Layer {layer}', fontsize=14)
                    ax.set_xlabel('Number of Repetitions', fontsize=12)
                    ax.set_ylabel('Mean TCAV Score', fontsize=12)
                    ax.legend(loc='upper left')

                    # Save the figure
                    fullpath = os.path.join(savefolder,
                                            f'mean_{filename}_subset_{subset_idx}_class_{class_id}_concept_{idx_concept}_layer_{layer}.png')
                    plt.tight_layout()
                    plt.savefig(fullpath)
                    plt.close()





    return


def track_mean_stabilization(grand_means_np, window=10):
    stabilization_index = []

    # Fixed stability threshold of 5%
    stability_threshold = 0.05

    # Start tracking from the beginning
    start_index = 0
    end_index = len(grand_means_np)

    # Iterate through grand means
    for i in range(start_index, end_index):
        # Get the window of means for the current and previous repetitions
        if i - window + 1 >= 0:
            window_means = grand_means_np[i - window + 1:i + 1]  # Extract window of means
            current_mean = grand_means_np[i]
            mean_of_window = np.mean(window_means)

            # Calculate the percentage change from the mean of the window
            percentage_change = abs(current_mean - mean_of_window) / mean_of_window

            # Check if the change is within the fixed threshold of 5%
            if percentage_change < stability_threshold:
                stabilization_index.append(i)  # Record stabilization index
            else:
                stabilization_index=[] #reset

    # Check if we have at least 10 stable indices and if it's continuous until the end
    if len(stabilization_index) >= 10:
        is_continuous = True
        return stabilization_index, is_continuous
    else:
        return [], False  # Return empty if not stable for at least 10 indices
def plot_cumulative_tcav_grandmean_for_class(savefolder, filename, layers, dict_of_stats, experimental_sets,
                                             num_elements, target_class):
    if target_class is None:
        class_list = range(5)
    else:
        class_list = target_class
    num_classes=len(class_list)

    # Define a color map for layers
    color_map = plt.get_cmap('tab10')  # You can choose any colormap you like

    for target_class in class_list:
        # Iterate through subsets
        for subset_idx, subset in enumerate(experimental_sets):
            # Create a new figure for each subset
            fig, ax = plt.subplots(figsize=(10, 6))


            ax.set_ylim([0, 1])
            text_content = []
            # Iterate through layers to plot both concepts for each layer in the current subset
            for layer_indx, layer in enumerate(layers):
                idx_concept = 0  # Assuming two concepts (0 and 1) #todo they are mirrored anyway
                # Initialize lists to store median and IQR values across repetitions


                grand_means = []
                grand_stds = []
                repetition_counts = list(range(1, num_elements + 1))

                # Collect median and IQR for each repetition
                for rep_num in repetition_counts:
                    layer_stats = dict_of_stats.get(subset_idx, {}).get(target_class, {}).get(layer, {}).get(
                        idx_concept, {}).get(rep_num, {})
                    all_vals = []
                    if layer_stats:  # Ensure stats exist for the current repetition
                        #means = layer_stats.get('means')
                        #stds = layer_stats.get('std')
                        vals = layer_stats.get('vals', [])

                        #repetition_means.append(means)
                        all_vals.append(vals)

                        # Calculate the grand mean and grand std for all values up to the current repetition
                        grand_mean = np.mean(all_vals)
                        grand_std = np.std(all_vals)

                        grand_means.append(grand_mean)
                        grand_stds.append(grand_std)

                grand_means_np=np.array(grand_means)
                grand_stds_np=np.array(grand_stds)

                # Track stabilization
                #stabilization_index, is_stable = track_mean_stabilization(grand_means_np)

                # Choose a color for the current layer
                color = color_map(layer_indx)

                # Plot the medians across repetitions for the current concept, overlaid for each layer
                ax.plot(range(len(repetition_counts)), grand_means_np, label=f'Layer {layer}, Concept {idx_concept}',color=color)
                ax.fill_between(range(len(repetition_counts)), grand_means_np - grand_stds_np,
                                grand_means_np + grand_stds_np, alpha=0.3,color=color)

                # # Highlight stabilization areas
                # if is_stable:
                #     first_stable_idx = stabilization_index[0]  # Get the first stabilization index
                #     ax.axvline(x=first_stable_idx, linestyle='--', alpha=0.5,
                #                color=color)  # +1 for 1-based indexing
                #
                #     # Annotate the stabilization point with an 'x'
                #     ax.annotate('x',
                #                 xy=(first_stable_idx, grand_means_np[first_stable_idx]),
                #                 color=color,
                #                 fontsize=12,
                #                 fontweight='bold',
                #                 ha='center',
                #                 va='bottom')  # Adjust 'va' as needed for positioning
                    #text_content.append(f'Less then 5% change for at list 10 points at Rep {first_stable_idx} for layer {layer_indx}')


            # # Positioning the textbox
            # textbox_str = '\n'.join(text_content)
            # # Add a textbox to the plot
            # ax.text(0.95, 0.95, textbox_str, transform=ax.transAxes, fontsize=10,
            #         verticalalignment='top', horizontalalignment='right',
            #         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

            # Add titles and labels for the combined plot (for the current subset)
            ax.set_title(f'Subset {subset_idx} for Class {target_class} (All Layers & Concepts)', fontsize=14)
            ax.set_xlabel('Number of Repetitions', fontsize=12)
            ax.set_ylabel('Median TCAV Score', fontsize=12)

            # Move the legend below the plot with entries on new lines
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), fancybox=True, shadow=True, ncol=1, fontsize=10)

            # Adjust layout to make room for the legend below
            plt.tight_layout(rect=[0, 0, 1, 0.9])  # Increase bottom space for legend

            # Save the overlay figure for this subset
            fullpath = os.path.join(savefolder, f'overlay_{filename}_subset_{subset_idx}_class_{target_class}.png')
            plt.savefig(fullpath, bbox_inches='tight')  # Ensures the legend is fully captured in the saved figure
            plt.close()

    return

def plot_cumulative_tcav_median_for_class(savefolder, filename, layers, dict_of_stats, experimental_sets,
                                          num_elements, target_class):
    if target_class is None:
        class_list = range(5)
    else:
        class_list = target_class

    for target_class in class_list:
        # Iterate through subsets
        for subset_idx, subset in enumerate(experimental_sets):
            # Create a new figure for each subset
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_ylim([0, 1])

            # Iterate through layers to plot both concepts for each layer in the current subset
            for layer in layers:
                for idx_concept in range(2):  # Assuming two concepts (0 and 1)
                    # Initialize lists to store median and IQR values across repetitions
                    repetition_medians = []
                    repetition_iqrs = []
                    repetition_counts = list(range(1, num_elements + 1))  # Repetition numbers

                    # Collect median and IQR for each repetition
                    for rep_num in repetition_counts:
                        layer_stats = dict_of_stats.get(subset_idx, {}).get(target_class, {}).get(layer, {}).get(
                            idx_concept, {}).get(rep_num, {})

                        if layer_stats:  # Ensure stats exist for the current repetition
                            vals = layer_stats.get('vals', [])
                            filtered_vals = remove_outliers(vals)

                            # Compute median and IQR
                            median = np.median(filtered_vals)
                            q1 = np.percentile(filtered_vals, 25)  # 25th percentile (Q1)
                            q3 = np.percentile(filtered_vals, 75)  # 75th percentile (Q3)
                            iqr = q3 - q1  # IQR

                            repetition_medians.append(median)
                            repetition_iqrs.append(iqr)
                        else:
                            repetition_medians.append(np.nan)
                            repetition_iqrs.append(np.nan)

                    # Convert to numpy arrays for easier plotting
                    repetition_medians = np.array(repetition_medians)
                    repetition_iqrs = np.array(repetition_iqrs)

                    # Calculate the confidence intervals
                    n = num_elements  # Number of repetitions
                    z_score = stats.norm.ppf(0.975)  # Z-score for 95% CI
                    ci_margin = z_score * (repetition_iqrs / np.sqrt(n)) / 1.35  # Margin of error for the CI

                    # Plot the medians across repetitions for the current concept, overlaid for each layer
                    ax.plot(repetition_counts, repetition_medians, label=f'Layer {layer}, Concept {idx_concept}')
                    ax.fill_between(repetition_counts, repetition_medians - ci_margin,
                                    repetition_medians + ci_margin, alpha=0.3)

            # Add titles and labels for the combined plot (for the current subset)
            ax.set_title(f'Subset {subset_idx} for Class {target_class} (All Layers & Concepts)', fontsize=14)
            ax.set_xlabel('Number of Repetitions', fontsize=12)
            ax.set_ylabel('Median TCAV Score', fontsize=12)

            # Move the legend below the plot with entries on new lines
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), fancybox=True, shadow=True, ncol=1, fontsize=10)

            # Adjust layout to make room for the legend below
            plt.tight_layout(rect=[0, 0, 1, 0.9])  # Increase bottom space for legend

            # Save the overlay figure for this subset
            fullpath = os.path.join(savefolder, f'overlay_{filename}_subset_{subset_idx}_class_{target_class}.png')
            plt.savefig(fullpath, bbox_inches='tight')  # Ensures the legend is fully captured in the saved figure
            plt.close()

    return

def plot_histogram(dict_of_stats, experimental_sets, pathname, filename, target_class):

    num_sets = len(experimental_sets)
    concepts = [0, 1]  # Assuming two concepts
    score_type="sign_count"

    if target_class is None:
        class_list = range(5)
    else:
        class_list = target_class

    colors = ['blue', 'orange']  # Colors for different concepts
    alphas = [0.6, 0.6]  # Transparency for overlapping histograms

    # Total grid size based on batching (2), classes (variable), and experimental sets (num_sets)
    num_layer = len(layers)  # Total number of classes

    for class_id in class_list:
        for idx_es, subset in enumerate(experimental_sets):  # Horizontal: experimental sets

            fig, axs = plt.subplots(1, num_layer, figsize=(15,5), sharex=True, sharey=True)

            for idx_layer, layer in enumerate(layers):
                ax = axs[idx_layer]  # Select correct subplot

                #ax.cla()

                # Plot Q-Q plot for each concept (on the same axis)
                for concept_idx,concept in enumerate(subset):

                    layer_stats = dict_of_stats.get(idx_es, {}).get(class_id, {}).get(layer, {}).get(concept_idx, {})
                    vals=layer_stats[len(layer_stats)]["vals"]

                    if vals is None:
                        print("vals empty")
                    if len(vals)==0:
                        print("vals of 0 lenght")

                    # Plot the Q-Q plot for each concept
                    vals=np.array(vals)
                    #bins=len(vals)
                    bins=math.ceil(math.sqrt(len(vals)))
                    ax.hist(vals, bins=bins, range=[0,1], alpha=alphas[concept_idx], color=colors[concept_idx], label=f'Concept {concept_idx}')
                    #sm.qqplot(vals, line='45',ax=ax)

                    # Set axis labels only for bottom row and first column plots for clarity
                    #if idx_es == num_sets - 1:
                    layer_label = layer.replace("densenet121.features.", "")
                    ax.set_xlabel(f'Layer {layer_label}')
                    #if idx_layer == 0:
                    ax.set_ylabel(f'Subset {idx_es}')

                    # Add legend to distinguish between concepts
                    ax.legend(loc='upper right')

                    ax.set_ylim([0,50])
                    #ax.set_aspect('equal')


                    # Set overall figure title and adjust layout
            fig.suptitle(f'Histogram for class {class_id} ', fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            # Save the figure
            fullpath = os.path.join(pathname, f'histogram {filename} for class {class_id} for subset {idx_es}.png')
            fig.savefig(fullpath)
            plt.close()

    return

def package_concepts(concepts_list, random_concept, healthy_concept):
    experimental_set_rand = []

    for concept in concepts_list:
        # Create a list of pairs for each concept
        concept_comparison = [
            [concept, random_concept],
            [concept, healthy_concept]
        ]

        # Append the comparison list to the main list
        experimental_set_rand.append(concept_comparison)

    return experimental_set_rand

def calc_all_concepts(dl_name,cav_folder,layers,experimental_sets_abs,repeat_nr,type_name,picklefolder,modelnr,data_loader,target_class):


    for experimental_set_abs in experimental_sets_abs:
        exp_name = experimental_set_abs[0][0].name
        abbr_value = exp_name.split("abbr-")[1].split("_exp")[0]
        if exp_name=="type-pos_abbr-BCoA_exp-2_form-polygon_resize-false_bg-original":
            abbr_value = "BCoA2"
        elif exp_name== "type-pos_abbr-BCoA_exp-5_form-polygon_resize-false_bg-original":
            abbr_value = "BCoA5"
        elif exp_name== "type-pos_abbr-BCoA_exp-1_form-polygon_resize-false_bg-synth":
            abbr_value = "BCoAsynth"


        run_repetition_wrapper(dl_name,cav_folder,layers,repeat_nr, experimental_set_abs, type_name, picklefolder, abbr_value, modelnr,data_loader,target_class)
    return

if __name__ == "__main__":

    import torch
    import numpy as np
    import random

    # Set seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32' #todo can remove after all
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # todo can remove after all
    torch.cuda.empty_cache()

    concepts_path = "/home/fkraehenbuehl/projects/SalCon/data/concepts"

    BCoA="type-pos_abbr-BCoA_exp-1_form-polygon_resize-false_bg-original"
    BCaA="type-pos_abbr-BCaA_exp-1_form-polygon_resize-false_bg-original"
    MSign="type-pos_abbr-MSign_exp-1_form-polygon_resize-false_bg-original"
    FIHOOF = "type-pos_abbr-FIHOOF_exp-1_form-polygon_resize-false_bg-original"
    AB = "type-pos_abbr-AB_exp-1_form-polygon_resize-false_bg-original"
    BO = "type-pos_abbr-BO_exp-1_form-polygon_resize-false_bg-original"
    KL = "type-pos_abbr-KL_exp-1_form-polygon_resize-false_bg-original"
    PC = "type-pos_abbr-PC_exp-1_form-polygon_resize-false_bg-original"
    PILi = "type-pos_abbr-PlLi_exp-1_form-polygon_resize-false_bg-original"
    SAS = "type-pos_abbr-SAS_exp-1_form-polygon_resize-false_bg-original"
    healthy_patches = "healthy_patches"
    random_patches = "random_patches"
    cable = "type-pos_abbr-cable_exp-1_form-polygon_resize-false_bg-original"
    Marker = "type-pos_abbr-marker_exp-1_form-polygon_resize-false_bg-original"
    BCoA_wo_test = "type-pos_abbr-BCoA_exp-1_form-polygon_resize-false_bg-original-wo-test"
    healthy_patches_valid= "healthy_patches_valid"
    random_patches_valid= "random_patches_valid"
    #BCoA_onefive="type-pos_abbr-BCoA_exp-1.5_form-polygon_resize-false_bg-original"
    BCoA_two = "type-pos_abbr-BCoA_exp-2_form-polygon_resize-false_bg-original"
    BCoA_five = "type-pos_abbr-BCoA_exp-5_form-polygon_resize-false_bg-original"
    #BCoA_square = "type-pos_abbr-BCoA_exp-1_form-square_resize-false_bg-original"
    BCoA_synth = "type-pos_abbr-BCoA_exp-1_form-polygon_resize-false_bg-synth"

    device = setup_device(4) #todo can be anything

    BCoA_concept = assemble_concept(BCoA, 0, device,concepts_path=concepts_path)
    BCoA_tmp_concept = assemble_concept(BCoA, 17, device, concepts_path=concepts_path) #todo debug
    BCaA_concept = assemble_concept(BCaA, 3, device, concepts_path=concepts_path) #todo ID changed for pickle
    MSign_concept = assemble_concept(MSign, 4, device,concepts_path=concepts_path) #todo ID changed for pickle
    FIHOOF_concept = assemble_concept(FIHOOF, 6, device, concepts_path=concepts_path)
    AB_concept = assemble_concept(AB, 7, device, concepts_path=concepts_path)
    BO_concept = assemble_concept(BO, 8, device, concepts_path=concepts_path)
    KL_concept = assemble_concept(KL, 9, device, concepts_path=concepts_path)
    PC_concept= assemble_concept(PC, 10, device, concepts_path=concepts_path)
    PILi_concept = assemble_concept(PILi, 11, device, concepts_path=concepts_path)
    SAS_concept = assemble_concept(SAS, 12, device, concepts_path=concepts_path)
    cable_concept = assemble_concept(cable, 13, device, concepts_path=concepts_path)
    Marker_concept = assemble_concept(Marker, 5, device, concepts_path=concepts_path)
    BCoA_wo_test_concept = assemble_concept(BCoA_wo_test, 14, device, concepts_path=concepts_path)
    #BCoA_onefive_concept = assemble_concept(BCoA_onefive, 0, device, concepts_path=concepts_path)
    BCoA_two_concept = assemble_concept(BCoA_two, 18, device, concepts_path=concepts_path)
    BCoA_five_concept = assemble_concept(BCoA_five, 19, device, concepts_path=concepts_path)
    #BCoA_square_concept = assemble_concept(BCoA_square, 20, device, concepts_path=concepts_path)
    BCoA_synth_concept = assemble_concept(BCoA_synth, 22, device, concepts_path=concepts_path)


    healthy_patches_concept = assemble_concept(healthy_patches, 2, concepts_path=concepts_path,device=device) #todo ID changed for pickle
    random_patches_concept = assemble_concept(random_patches, 1, concepts_path=concepts_path,device=device) #todo ID changed for pickle
    healthy_patches_valid_concept = assemble_concept(healthy_patches_valid, 16, concepts_path=concepts_path,device=device) #todo ID changed for pickle
    random_patches_valid_concept = assemble_concept(random_patches_valid, 15, concepts_path=concepts_path,device=device) #todo ID changed for pickle

    path_load_model_0 = '/home/fkraehenbuehl/projects/SalCon/model/models/model_0/densenet pretrain unweighted bce with class weight wd0.0001_model_gc_lr0.0001_epoches5.pt'
    path_load_model_1 = '/home/fkraehenbuehl/projects/SalCon/model/models/model_1/densenet_model_bilinear_lr0.0001_epoches5.pt'
    path_load_model_2 = '/home/fkraehenbuehl/projects/SalCon/model/models/model_2/densenet_model_bilinear_lr0.0001_epoches5-1.pt'
    path_load_model_3 = '/home/fkraehenbuehl/projects/SalCon/model/models/model_3/markermodel_model_bilinear_lr0.0001_epoches5.pt'

    modelnr=1
    if modelnr==0:
        path_load_model = '/home/fkraehenbuehl/projects/SalCon/model/models/model_0/densenet pretrain unweighted bce with class weight wd0.0001_model_gc_lr0.0001_epoches5.pt'
    elif modelnr==1:
        path_load_model = '/home/fkraehenbuehl/projects/CaptumTCAV/prep-model/models/w_concepts_model_bilinear_lr0.0001_epoches5.pt'
    elif modelnr==2:
        path_load_model = '/home/fkraehenbuehl/projects/CaptumTCAV/prep-model/models/wo_concepts_model_bilinear_lr0.0001_epoches5.pt'
    elif modelnr==3:
        path_load_model ='/home/fkraehenbuehl/projects/CaptumTCAV/prep-model/models/w_marker_model_bilinear_lr0.0001_epoches5.pt'

    model=load_and_prepare_model(path_load_model, 5, device)

    layers = [
    "densenet121.features.denseblock1.denselayer6.conv2",
    "densenet121.features.denseblock2.denselayer12.conv2",
    "densenet121.features.denseblock3.denselayer24.conv2",
    "densenet121.features.denseblock4.denselayer16.conv2"]

    # Load the dataloader
    dl_name="test" #test makes more sense
    if dl_name=='train':
        dataloader = load_data(dataset_type="train")
    if dl_name=='valid':
        dataloader = load_data(dataset_type="valid")
    if dl_name=='test':
        dataloader = load_data(dataset_type="test")
    if dl_name=='testwmarker':
        dataloader = load_data(dataset_type="testwmarker")

    batching=True

    figurefolder= "./chexpert-figures-3" #todo debug 2
    if not os.path.exists(figurefolder):
        os.makedirs(figurefolder)

    picklefolder = "./chexpert-pickles-3" #todo debug 2
    if not os.path.exists(picklefolder):
        os.makedirs(picklefolder)

    cav_folder = "./chexpert-cav-3" #todo debug 2
    if not os.path.exists(cav_folder):
        os.makedirs(cav_folder)

    #run absolute comparision
    repeat_nr = 24
    type_name=f"abs"


    # todo took out debug BCoA,BCaA, because already calculated
    # concepts = [BCoA_concept, BCaA_concept,MSign_concept, FIHOOF_concept, AB_concept, BO_concept, KL_concept, PC_concept, PILi_concept,
    #             SAS_concept, cable_concept, Marker_concept]
    # todo debug for now
    #concepts = [BCoA_concept, BCaA_concept,MSign_concept, FIHOOF_concept, AB_concept]
    #concepts = [PILi_concept,SAS_concept]
    #BCoA_onefive_concept,BCoA_two_concept,BCoA_five_concept,BCoA_square_concept,BCoA_synth_concept
    concepts = [BCoA_concept]#, PILi_concept,SAS_concept]
    target_class=[3,4]#[0,1,2,3,4]
    experimental_sets_abs = package_concepts(concepts, random_patches_concept, healthy_patches_concept)
    calc_all_concepts(dl_name,cav_folder,layers,experimental_sets_abs,repeat_nr,type_name,picklefolder,modelnr,dataloader,target_class) #todo debug

    type_filter = f"abs"  # todo important to not forget!!!
    model_filter = modelnr
    repetition=None
    score_type="sign_count"

    for experimental_set_abs in experimental_sets_abs:
        exp_name = experimental_set_abs[0][0].name
        abbr_value = exp_name.split("abbr-")[1].split("_exp")[0]
        if exp_name == "type-pos_abbr-BCoA_exp-2_form-polygon_resize-false_bg-original":
            abbr_value = "BCoA2"
        elif exp_name == "type-pos_abbr-BCoA_exp-5_form-polygon_resize-false_bg-original":
            abbr_value = "BCoA5"
        elif exp_name == "type-pos_abbr-BCoA_exp-1_form-polygon_resize-false_bg-synth":
            abbr_value = "BCoAsynth"

        name_filter = f"{abbr_value}"

        #barplot for means
        dict_of_stats,num_elements=calculate_tcav_stats(model_filter,dl_name,name_filter,type_filter,picklefolder, layers, experimental_set_abs, score_type="sign_count", repetition=None)
        filename=f"meanplot_name_{name_filter}_model_{model_filter}_ds_{dl_name}"
        plot_tcav_scores_per_class_mean(figurefolder, filename, layers, dict_of_stats,
                                   experimental_set_abs, num_elements,target_class)
        # filename = f"medianplot_name_{name_filter}_model_{model_filter}_ds_{dl_name}"
        # plot_tcav_scores_per_class_median(figurefolder, filename, layers, dict_of_stats,
        #                            experimental_set_abs,num_elements,target_class,name_filter)
        filename = f"significance_{name_filter}_model_{model_filter}_ds_{dl_name}"
        calc_significance(figurefolder, filename, layers, dict_of_stats,
                                   experimental_set_abs,num_elements,target_class)


        #line plot tracking means
        dict_of_stats,num_elements=calculate_tcav_stats_with_repetitions(model_filter,dl_name,name_filter,type_filter,picklefolder, layers, experimental_set_abs, score_type="sign_count", repetition=None)
        # filename = f"cumulative_median_name_{name_filter}_model_{model_filter}"
        # plot_cumulative_tcav_median_for_class(figurefolder, filename, layers, dict_of_stats,
        #                                       experimental_set_abs, num_elements, target_class)
        filename = f"cumulative_mean_name_{name_filter}_model_{model_filter}_ds_{dl_name}"
        plot_cumulative_tcav_grandmean_for_class(figurefolder, filename, layers, dict_of_stats,
                                                 experimental_set_abs, num_elements, target_class)

        filename = f"histogram_name_{name_filter}_model_{model_filter}_ds_{dl_name}"
        plot_histogram(dict_of_stats, experimental_set_abs, figurefolder, filename, target_class)

        print(f'plotted all for {name_filter}')


    print("ran trough script")