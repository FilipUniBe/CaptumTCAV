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

    tcav_scores_w_random = mytcav.interpret(inputs=zebra_tensors,
                                            experimental_sets=experimental_set_rand,
                                            target=zebra_ind,
                                            n_steps=5,
                                            )
    filename="absolute_TCAV.jpg"
    plot_tcav_scores(experimental_set_rand, tcav_scores_w_random,filename)

    experimental_set_zig_dot = [[stripes_concept, zigzagged_concept, dotted_concept]]

    tcav_scores_w_zig_dot = mytcav.interpret(inputs=zebra_tensors,
                                             experimental_sets=experimental_set_zig_dot,
                                             target=zebra_ind,
                                             n_steps=5)
    filename = "relative_TCAV.jpg"
    plot_tcav_scores(experimental_set_zig_dot, tcav_scores_w_zig_dot,filename)

    print("Script Finished")