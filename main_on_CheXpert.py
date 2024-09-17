#conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch
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

from scipy.stats import ttest_ind

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

def plot_tcav_scores_grid(experimental_sets, savefolder,layers, filename,batching_filter,model_filter,name_filter,classes):
    repetitions=3
    classes_list = range(classes)
    repetitions_list = range(repetitions)
    layer_labels = generate_layer_labels(layers)

    for subset_idx,subset in enumerate(experimental_sets):

        # Create the figure with a grid of subplots
        fig, axs = plt.subplots(len(repetitions_list), len(classes_list), figsize=(20, 15))

        barWidth = 0.15  # Adjusted width to reduce overlap
        spacing = 0.2  # Add spacing between groups of bars

        for idx_class, class_id in enumerate(classes_list):
            for idx_rep, repetition in enumerate(repetitions_list):

                all_tcav_scores,_ = load_and_filter_pickles(savefolder, idx_class, batching_filter,
                                                                          idx_rep, model_filter, name_filter)
                relevant_scores=all_tcav_scores[0]#must be just one though

                _ax = axs[idx_rep, idx_class]  # Use appropriate subplot

                concepts = subset
                concepts_key = concepts_to_str(concepts)

                pos = np.arange(len(layers))  # Base position for bars
                bar_positions = [pos + i * (barWidth + spacing) for i in range(len(concepts))]

                for i in range(len(concepts)):
                    adjusted_pos = bar_positions[i]
                    val = [format_float(scores['sign_count'][i]) for layer, scores in relevant_scores[concepts_key].items()]
                    _ax.bar(adjusted_pos, val, width=barWidth, edgecolor='white', label=concepts[i].name)

                # Add xticks on the middle of the group bars
                _ax.set_title(f'Class {class_id}, Repetition {repetition}', fontsize=12)
                _ax.set_xticks(pos + (len(concepts) - 1) * (barWidth + spacing) / 2)
                _ax.set_xticklabels(layer_labels, fontsize=10)
                _ax.set_xlabel('Layers')

        # Add a single legend for all subplots
        handles, labels = _ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=len(subset), bbox_to_anchor=(0.5, 0.1),
                   bbox_transform=fig.transFigure, fontsize=10)

        # Adjust layout and save the figure
        plt.tight_layout()
        modified_filename=f'{filename}_subset_{subset_idx}'
        fullpath = os.path.join(savefolder, modified_filename)
        plt.suptitle(f'TCAV Scores for {modified_filename}', fontsize=16)
        plt.savefig(fullpath)
        plt.close()

    return


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
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}. Choose from 'valid', 'test', or 'train'.")

    # load CSV
    test_labels_meta = pd.read_csv(csv_file)

    # Get the list of columns to drop
    columns_to_drop = test_labels_meta.columns.difference(exist_labels + ["Path"])

    # Drop the columns not in the specified list
    test_labels_meta = test_labels_meta.drop(columns=columns_to_drop)

    # Check if all columns in the specified list have a value of 0 for each row
    mask = test_labels_meta[exist_labels].eq(0).all(axis=1)

    # Drop the rows where all columns have a value of 0
    test_labels_meta = test_labels_meta[~mask]
    print("Amount of images used: ", len(test_labels_meta.index))

    # Exclude file paths where the last part ends with "_latera.jpg"
    test_labels_meta = test_labels_meta[
        ~test_labels_meta["Path"].str.endswith("_lateral.jpg") #todo not test exclusively anymore
    ]

    # values only
    y_test_all = test_labels_meta[exist_labels].values

    # paths only
    x_test_path = test_labels_meta.Path.values

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
        checkpoint = torch.load(model, map_location=device)

    model = DenseNet121(num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
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

def BCOA_from_pickle():
    target_class=3
    pickle_file="/home/fkraehenbuehl/projects/SalCon/results/experiment_019_20240916-154625/combined_data.pkl"
    with open(pickle_file, 'rb') as f:
        combined_data = pickle.load(f)
    tcav_score = combined_data['tcav_score']

    filename=f"CheXpert_absolute_TCAV_class_{target_class}_model_{modelnr}_from_pickle.jpg"
    plot_tcav_scores(experimental_set_rand, tcav_score, filename)
    print("plotted")
    print("finished")
    return

def run_TCAV_target_class_wrapper(experimental_set,mytcav,name, savefolder, target_class=None, batching=True):
    if target_class is None:
        for target_class in range(5):
            filename = f"CheXpert_{name}_class_{target_class}_model_{modelnr}_batching_{batching}.jpg"
            # If pickles exist, skip calculation and load results
            pickle_path = os.path.join(savefolder, f'{filename}.pkl')
            if os.path.exists(pickle_path):
                print("using pickle")
                with open(pickle_path, 'rb') as f:
                    mean_scores = pickle.load(f)
            else:
                print("calculating from scratch")
                mean_scores=run_TCAV(target_class,experimental_set,mytcav,batching)
                with open(pickle_path, 'wb') as f:
                    pickle.dump(mean_scores, f)

            filename = f"CheXpert_{name}_class_{target_class}_model_{modelnr}_batching_{batching}.jpg"
            plot_tcav_scores(experimental_set, savefolder, mean_scores, filename)
            print(f"saved {filename}")

    return

def run_TCAV(target_class,experimental_set,mytcav,batching=True):

    tcav_scores_all_batches=[]
    for images, _, _ in tqdm(test_dataloader, desc="Loading images into memory"):
        images=images.to(device)
        tcav_scores_w_random = mytcav.interpret(inputs=images,
                                                experimental_sets=experimental_set,
                                                target=target_class,
                                                n_steps=5,
                                                )

        tcav_scores_all_batches.append(tcav_scores_w_random)
        if not batching:
            break #stop after one batch
    return mean_score(tcav_scores_all_batches)

def run_repetition_wrapper(repeat_nr,experimental_set,name):
    for current_repeat in tqdm(range(repeat_nr), desc="repeating calculation n times"):
        mytcav = TCAV(model=model,
                      layers=layers,
                      layer_attr_method=LayerIntegratedGradients(
                          model, None, multiply_by_inputs=False),
                      save_path=f"./cav-model-{modelnr}-repeat-{current_repeat}/")

        run_TCAV_target_class_wrapper(experimental_set,mytcav,f'name_{name}_rep_{current_repeat}', savefolder, target_class,
                                      batching)
    return


def load_and_filter_pickles(directory, class_filter=None, batching_filter=None, repetition_filter=None,model_filter=None,name_filter=None):
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
        name_match = re.search(r'name_([^_]+)', file)

        # Filter based on provided arguments
        class_cond = class_filter is None or (class_match and int(class_match.group(1)) == class_filter)
        batching_cond = batching_filter is None or (batching_match and batching_match.group(1) == str(batching_filter))
        repetition_cond = repetition_filter is None or (
                    repetition_match and int(repetition_match.group(1)) == repetition_filter)
        model_cond = model_filter is None or (model_match and int(model_match.group(1)) == model_filter)
        name_cond = name_filter is None or (name_match and name_match.group(1) == name_filter)

        if class_cond and batching_cond and repetition_cond and model_cond and name_cond:
            filtered_files.append(file)

    all_tcav_scores=[]
    for file in filtered_files:
        with open(os.path.join(directory, file), 'rb') as f:
            mean_scores = pickle.load(f)
            all_tcav_scores.append(mean_scores)
            #todo should be just one though!

    return all_tcav_scores,filtered_files


def calculate_mean_std(savefolder, layers, classes, repetitions, batching_filter, model_filter, name_filter, experimental_sets):



    dict_of_stats={}
    for subset_idx, subset in enumerate(experimental_sets):
        dict_of_stats[subset_idx] = {}  # Initialize dictionary for this subset
        concepts_key = concepts_to_str(subset)
        for class_id in range(classes):
            dict_of_stats[subset_idx][class_id] = {}  # Initialize dictionary for this class

            for layer in layers:

                dict_of_stats[subset_idx][class_id][layer] = {}  # Initialize dictionary for this class
                for idx in range(2):
                    dict_of_stats[subset_idx][class_id][layer][idx] = {}  # Initialize dictionary for this class
                    values_list = []
                    for repetition in range(repetitions):
                        all_tcav_scores, _ = load_and_filter_pickles(savefolder, class_id, batching_filter, repetition,
                                                                     model_filter, name_filter)
                        relevant_scores = all_tcav_scores[0]  # Assuming there's just one set of scores

                        # Extract values for the current layer and convert to numpy array
                        values_list.append(relevant_scores[concepts_key][layer]['sign_count'][idx])
                        values_list = [value.cpu() for value in values_list]

                    means = np.mean(values_list, axis=0)
                    std = np.std(values_list, axis=0)


                    # Store results in the dictionary
                    dict_of_stats[subset_idx][class_id][layer][idx] = {
                        "means": means,
                        "std": std
                    }

    return dict_of_stats


def plot_tcav_scores_per_class(savefolder, filename, layers, dict_stats, experimental_sets,repeat_nr):

    classes_list=range(5)
    num_classes=len(classes_list)

    for subset_idx, subset in enumerate(experimental_sets):
        fig, axs = plt.subplots(num_classes, 1, figsize=(15, 5 * num_classes), sharex=True)

        barWidth = 0.15
        spacing = 0.2  # Spacing between bars for different concepts
        layer_positions = np.arange(len(layers))

        for idx_class, class_id in enumerate(classes_list):
            # Plot mean and std with error bars for each class

            _ax = axs[idx_class]  # Use appropriate subplot
            concepts=subset

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
                _ax.bar(adjusted_pos, means, width=barWidth, yerr=stds, capsize=5,
                       edgecolor='white', label=f'{concepts[i]}')

            _ax.set_title(f'Class {class_id}', fontsize=14)
            _ax.set_ylabel('Mean TCAV Score', fontsize=12)
            _ax.set_xticks(layer_positions + (len(subset) - 1) * (barWidth + spacing) / 2)
            _ax.set_xticklabels(layers, fontsize=10)
            _ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=len(subset), fontsize=10)

        fig.suptitle(f'TCAV Scores for Subset {subset_idx} random: {repeat_nr}', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit suptitle

        # Save the figure
        fullpath = os.path.join(savefolder, f'{filename}_subset_{subset_idx}.png')
        plt.savefig(fullpath)
        plt.close()

    return

def calculate_tcav_stats(savefolder, layers, experimental_set_rand, score_type="sign_count", repetition=None):

    # Use defaultdict to avoid manual initialization of dictionaries
    dict_of_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

    for subset_idx, subset in enumerate(experimental_set_rand):

        for class_id in range(5):

            for score_layer in layers:

                all_tcav_scores, _ = load_and_filter_pickles(savefolder, class_id, batching_filter, repetition,
                                                             model_filter, name_filter)
                for idx in range(2):

                    score_list = [
                        tcav_score["-".join([str(c.id) for c in subset])][score_layer][score_type][idx].cpu()
                        for tcav_score in all_tcav_scores
                    ]

                    means = np.mean(score_list, axis=0)
                    std = np.std(score_list, axis=0)

                    dict_of_stats[subset_idx][class_id][score_layer][idx]={"means": means, "std":std}
    return dict_of_stats


def calculate_tcav_stats_with_repetitions(savefolder, layers, experimental_set_rand, score_type="sign_count",repetition=None
                                          ):
    # Use defaultdict to avoid manual initialization of dictionaries
    dict_of_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))))

    for subset_idx, subset in enumerate(experimental_set_rand):

        for class_id in range(5):

                for score_layer in layers:

                    all_tcav_scores, _ = load_and_filter_pickles(savefolder, class_id, batching_filter, repetition,
                                                                 model_filter, name_filter)
                    # Iterate over the number of repetitions
                    for rep_num in range(1, len(all_tcav_scores) + 1):
                        for idx in range(2):
                            score_list = [
                                tcav_score["-".join([str(c.id) for c in subset])][score_layer][score_type][idx].cpu()
                                for tcav_score in all_tcav_scores[:rep_num]
                            ]

                            means = np.mean(score_list, axis=0)
                            std = np.std(score_list, axis=0)

                            dict_of_stats[subset_idx][class_id][score_layer][idx][rep_num] = {"means": means, "std": std}
    return dict_of_stats


def plot_tcav_means_across_repetitions(savefolder, filename, layers, dict_of_stats, experimental_sets,repeat_nr):
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
                    repetition_counts = list(range(1, repeat_nr + 1))  # Repetition numbers

                    for rep_num in repetition_counts:
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

if __name__ == "__main__":

    import torch
    import numpy as np
    import random

    # Set seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32' #todo can remove after all

    concepts_path = "/home/fkraehenbuehl/projects/SalCon/data/concepts"

    BCoA="type-pos_abbr-BCoA_exp-1_form-polygon_resize-false_bg-original"
    BCaA="type-pos_abbr-BCaA_exp-1_form-polygon_resize-false_bg-original"
    MSign="type-pos_abbr-MSign_exp-1_form-polygon_resize-false_bg-original"
    FIHOOF = "type-pos_abbr-FIHOOF_exp-1_form-polygon_resize-false_bg-original"
    AB = "type-pos_abbr-AB_exp-1_form-polygon_resize-false_bg-original"
    BO = "type-pos_abbr-BO_exp-1_form-polygon_resize-false_bg-original"
    KL = "type-pos_abbr-KL_exp-1_form-polygon_resize-false_bg-original"
    PC = "type-pos_abbr-PC_exp-1_form-polygon_resize-false_bg-original"
    PILi = "type-pos_abbr-PILi_exp-1_form-polygon_resize-false_bg-original"
    SAS = "type-pos_abbr-SAS_exp-1_form-polygon_resize-false_bg-original"
    healthy_patches = "healthy_patches"
    random_patches = "random_patches"
    cable = "type-pos_abbr-cable_exp-1_form-polygon_resize-false_bg-original"
    Marker = "type-pos_abbr-Marker_exp-1_form-polygon_resize-false_bg-original"

    device = setup_device(4)

    BCoA_concept = assemble_concept(BCoA, 0, device,concepts_path=concepts_path)
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

    healthy_patches_concept = assemble_concept(healthy_patches, 2, concepts_path=concepts_path,device=device) #todo ID changed for pickle
    random_patches_concept = assemble_concept(random_patches, 1, concepts_path=concepts_path,device=device) #todo ID changed for pickle

    path_load_model_0 = '/home/fkraehenbuehl/projects/SalCon/model/models/model_0/densenet pretrain unweighted bce with class weight wd0.0001_model_gc_lr0.0001_epoches5.pt'
    path_load_model_1 = '/home/fkraehenbuehl/projects/SalCon/model/models/model_1/densenet_model_bilinear_lr0.0001_epoches5.pt'
    path_load_model_2 = '/home/fkraehenbuehl/projects/SalCon/model/models/model_2/densenet_model_bilinear_lr0.0001_epoches5-1.pt'
    path_load_model_3 = '/home/fkraehenbuehl/projects/SalCon/model/models/model_3/markermodel_model_bilinear_lr0.0001_epoches5.pt'

    path_load_model=path_load_model_0
    if "0" in path_load_model:
        modelnr=0
    elif "1" in path_load_model:
        modelnr=1
    elif "2" in path_load_model:
        modelnr=2
    elif "3" in path_load_model:
        modelnr=3

    model=load_and_prepare_model(path_load_model, 5, device)

    layers = [
    "densenet121.features.denseblock1.denselayer6.conv2",
    "densenet121.features.denseblock2.denselayer12.conv2",
    "densenet121.features.denseblock3.denselayer24.conv2",
    "densenet121.features.denseblock4.denselayer16.conv2"]

    # Load the dataloader
    test_dataloader = load_data(dataset_type="test")

    experimental_set_rand = [[BCoA_concept, random_patches_concept], [BCoA_concept, healthy_patches_concept]]
    experimental_set_zig_dot = [[BCoA_concept, BCaA_concept, MSign_concept]]

    target_class=None
    batching=True

    savefolder="./chexpert-figures"
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)

    #run absolute comparision
    repeat_nr = 14#50
    name=f"abs"
    #run_repetition_wrapper(repeat_nr,experimental_set_rand,name) #todo disabled for debug
    # name = f"rel"
    # run_repetition_wrapper(repeat_nr,experimental_set_zig_dot,name) #todo disabled for debug


    name_filter = "abs"
    batching_filter = True
    model_filter = None
    filename=f"gridplot_name_{name_filter}_batching_{batching_filter}_model_{model_filter}"
    classes=5

    # plot gridplot for the repetitions (first 3)
    plot_tcav_scores_grid(experimental_set_rand, savefolder,layers, filename,batching_filter,model_filter,name_filter,classes)

    repetition=None
    score_type="sign_count"

    #barplot for means
    dict_of_stats=calculate_tcav_stats(savefolder, layers, experimental_set_rand, score_type="sign_count", repetition=None)
    filename=f"meanplot_name_{name_filter}_batching_{batching_filter}_model_{model_filter}"
    plot_tcav_scores_per_class(savefolder, filename, layers, dict_of_stats,
                               experimental_set_rand,repeat_nr)

    #line plot tracking means
    dict_of_stats=calculate_tcav_stats_with_repetitions(savefolder, layers, experimental_set_rand, score_type="sign_count", repetition=None)
    filename = f"trackingmean_name_{name_filter}_batching_{batching_filter}_model_{model_filter}"
    plot_tcav_means_across_repetitions(savefolder, filename, layers, dict_of_stats,
                               experimental_set_rand, repeat_nr)


    print("ran trough script")