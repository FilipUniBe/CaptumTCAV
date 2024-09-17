#conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch


import numpy as np
import os, glob

import matplotlib.pyplot as plt

from PIL import Image

from scipy.stats import ttest_ind

# ..........torch imports............
import torch
import torchvision

from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms

#.... Captum imports..................
from captum.attr import LayerGradientXActivation, LayerIntegratedGradients

from captum.concept import TCAV
from captum.concept import Concept

from captum.concept._utils.data_iterator import dataset_to_dataloader, CustomIterableDataset
from captum.concept._utils.common import concepts_to_str

# Method to normalize an image to Imagenet mean and standard deviation
def transform(img):

    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )(img)


def get_tensor_from_filename(filename):
    img = Image.open(filename).convert("RGB")
    return transform(img)

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

def load_image_tensors(class_name, root_path="data/tcav/image/concepts/", transform=True): #todo root_path adapted
    path = os.path.join(root_path, class_name)
    filenames = glob.glob(path + '/*.jpg')

    tensors = []
    for filename in filenames:
        img = Image.open(filename).convert('RGB')
        tensors.append(transform(img) if transform else img)

    return tensors

def assemble_concept(name, id, concepts_path="data/tcav/image/concepts/"):
    concept_path = os.path.join(concepts_path, name) + "/"
    dataset = CustomIterableDataset(get_tensor_from_filename, concept_path)
    concept_iter = dataset_to_dataloader(dataset)

    return Concept(id=id, name=name, data_iter=concept_iter)

def format_float(f):
    return float('{:.3f}'.format(f) if abs(f) >= 0.0005 else '{:.3e}'.format(f))

def plot_tcav_scores(experimental_sets, tcav_scores,filename,pathname):
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

    fullname=os.path.join(pathname,filename)
    plt.savefig(fullname)
    plt.close()
    return

def batch_data(tensor_data, batch_size):
    """
    Splits a tensor into batches of specified batch size.
    Args:
        tensor_data (Tensor): The tensor to be split into batches.
        batch_size (int): The size of each batch.
    Returns:
        List of tensors where each tensor is a batch.
    """
    return [tensor_data[i:i + batch_size] for i in range(0, len(tensor_data), batch_size)]

if __name__ == "__main__":

    concepts_path = "data/tcav/image/concepts/"

    stripes_concept = assemble_concept("striped", 0, concepts_path=concepts_path)
    zigzagged_concept = assemble_concept("zigzagged", 1, concepts_path=concepts_path)
    dotted_concept = assemble_concept("dotted", 2, concepts_path=concepts_path)

    random_0_concept = assemble_concept("random_0", 3, concepts_path=concepts_path)
    random_1_concept = assemble_concept("random_1", 4, concepts_path=concepts_path)

    model = torchvision.models.googlenet(pretrained=True)
    model = model.eval()

    layers = ['inception4c', 'inception4d', 'inception4e']

    mytcav = TCAV(model=model,
                  layers=layers,
                  layer_attr_method=LayerIntegratedGradients(
                      model, None, multiply_by_inputs=False))

    experimental_set_rand = [[stripes_concept, random_0_concept], [stripes_concept, random_1_concept]]

    # Load sample images from folder
    zebra_imgs = load_image_tensors('zebra', transform=False)

    # Load sample images from folder
    zebra_tensors = torch.stack([transform(img) for img in zebra_imgs])
    zebra_ind = 340
    hippo_ind = 344
    index_list=[zebra_ind,hippo_ind]

    pathname='./zebra-figures'
    if not os.path.exists(pathname):
        os.makedirs(pathname)

    repetition_nr=3
    for repetition in range(repetition_nr):

        mytcav = TCAV(model=model,
                      layers=layers,
                      layer_attr_method=LayerIntegratedGradients(
                          model, None, multiply_by_inputs=False),
                      save_path=f"./cav-zebra-repeat-{repetition}/")
        for indexes in index_list:

            # Specify batch size (adjust based on your system's memory capacity)
            batch_size = 4  # Example batch size; modify according to your GPU memory limits
            # Apply batching
            batches = batch_data(zebra_tensors, batch_size)
            # Initialize a list to store TCAV scores for all batches
            all_tcav_scores = []

            # Iterate over each batch and run TCAV interpretation
            for batch in batches:
                tcav_scores_batch = mytcav.interpret(
                    inputs=batch,
                    experimental_sets=experimental_set_rand,
                    target=indexes,
                    n_steps=5
                )
                # Collect the results from each batch
                all_tcav_scores.append(tcav_scores_batch)
            tcav_scores_w_random=mean_score(all_tcav_scores)

            filename=f"absolute_TCAV_batching_true_ind_{indexes}_repetition_{repetition}.jpg"

            plot_tcav_scores(experimental_set_rand, tcav_scores_w_random,filename,pathname)

            experimental_set_zig_dot = [[stripes_concept, zigzagged_concept, dotted_concept]]

            tcav_scores_w_zig_dot = mytcav.interpret(inputs=zebra_tensors,
                                                     experimental_sets=experimental_set_zig_dot,
                                                     target=indexes,
                                                     n_steps=5)
            filename = f"relative_TCAV_batching_true_ind_{indexes}_repetition_{repetition}.jpg"
            plot_tcav_scores(experimental_set_zig_dot, tcav_scores_w_zig_dot,filename,pathname)

            print("Script Finished")