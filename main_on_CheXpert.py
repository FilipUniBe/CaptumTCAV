#conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch
import pickle
import subprocess
import sys
from functools import partial

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

def plot_tcav_scores(experimental_sets, tcav_scores,filename):
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

    plt.savefig(filename)
    plt.close()
    return

if __name__ == "__main__":

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

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
    MSign_concept = assemble_concept(MSign, 3, device,concepts_path=concepts_path) #todo ID changed for pickle
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



    mytcav = TCAV(model=model,
                  layers=layers,
                  layer_attr_method=LayerIntegratedGradients(
                      model, None, multiply_by_inputs=False),save_path=f"./cav-model-{modelnr}/")

    #experimental_set_rand = [[BCoA_concept, healthy_patches_concept], [BCoA_concept, random_patches_concept]]
    experimental_set_rand = [[BCoA_concept, random_patches_concept], [BCoA_concept, healthy_patches_concept]] #todo changed for pickle


    # Load sample images from folder
    # Load the dataloader
    test_dataloader = load_data(dataset_type="test")

    print("starting tcav.interpret()")

    def BCOA():
        for target_class in range(5):
            print(target_class)
            for images, _, _ in tqdm(test_dataloader, desc="Loading images into memory"):
                images=images.to(device)
                tcav_scores_w_random = mytcav.interpret(inputs=images,
                                                        experimental_sets=experimental_set_rand,
                                                        target=target_class,
                                                        n_steps=5,
                                                        )
                break #stop after one batch

            filename=f"CheXpert_absolute_TCAV_class_{target_class}_model_{modelnr}.jpg"
            plot_tcav_scores(experimental_set_rand, tcav_scores_w_random, filename)
            print("plotted")
            print("finished")
        return



    #BCOA()


    def BCOA_BCAA_MSIGN():
        experimental_set_zig_dot = [[BCoA_concept, BCaA_concept, MSign_concept]]

        target_class=4

        for images, _, _ in tqdm(test_dataloader, desc="Loading images into memory"):
            images = images.to(device)
            tcav_scores_w_random = mytcav.interpret(inputs=images,
                                                    experimental_sets=experimental_set_zig_dot,
                                                    target=target_class,
                                                    n_steps=5,
                                                    )
            break  # stop after one batch

        filename = f"CheXpert_relative_TCAV_class_{target_class}_model_{modelnr}.jpg"
        plot_tcav_scores(experimental_set_zig_dot, tcav_scores_w_random,filename)

        print("Script Finished")
        return

    #BCOA_BCAA_MSIGN()

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

    BCOA_from_pickle()