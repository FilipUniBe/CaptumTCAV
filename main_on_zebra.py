#conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch
import math
import pickle
from collections import defaultdict

import numpy as np
import os, glob
import re
import itertools
import statsmodels.api as sm

import matplotlib.pyplot as plt

from PIL import Image

from scipy.stats import ttest_ind, ttest_ind_from_stats
from scipy import stats

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
from tqdm import tqdm

def get_model_size(model):
    # Calculate the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Assuming parameters are stored as float32 (4 bytes each)
    param_size_in_bytes = total_params * 4
    param_size_in_MB = param_size_in_bytes / (1024 ** 2)  # Convert to megabytes (MB)

    return param_size_in_MB

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


def TCAV_batched(zebra_tensors,indexes,mytcav,experimental_set,batch_size=4):
    # Apply batching
    batches = batch_data(zebra_tensors, batch_size)
    # Initialize a list to store TCAV scores for all batches
    all_tcav_scores = []

    # Iterate over each batch and run TCAV interpretation
    for batch in batches:
        tcav_scores_batch = mytcav.interpret(
            inputs=batch,
            experimental_sets=experimental_set,
            target=indexes,
            n_steps=5
        )
        # Collect the results from each batch
        all_tcav_scores.append(tcav_scores_batch)
    mean_score(all_tcav_scores)
    return mean_score(all_tcav_scores)

def calculate_tcav(name, experimental_set,picklefolder,pathname,repetition_nr,index_list, batch_size):

    for i in range(2):
        batching = (i == 0)  # True for i=0, False for i=1
        print(f'batching={batching}')

        for target_class in index_list:
            print(f'index={target_class}')

            all_tcav_scores = []
            for repetition in tqdm(range(repetition_nr), desc="calculating repetitinos"):

                mytcav = TCAV(model=model,
                              layers=layers,
                              layer_attr_method=LayerIntegratedGradients(
                                  model, None, multiply_by_inputs=False),
                              save_path=f"./zebra-cav/cav-zebra-repeat-{repetition}-batching-{batching}/")


                if batching == True:
                    tcav_scores = TCAV_batched(zebra_tensors,target_class,mytcav,experimental_set, batch_size)
                else:
                    lambda_tcav_scores = mytcav.interpret(
                        inputs=zebra_tensors,
                        experimental_sets=experimental_set,
                        target=target_class,
                        n_steps=5
                    )
                    # "De-lambda-ify" the dict (will otherwise crash pickleling)
                    tcav_scores = {}
                    for concept_key,concept in lambda_tcav_scores.items():
                        tcav_scores[concept_key]={}
                        for layer_key,layer in concept.items():
                            tcav_scores[concept_key][layer_key] = {}
                            for key,value in layer.items():
                                tcav_scores[concept_key][layer_key][key] = value.detach().cpu()

                filename = f"{name}_TCAV_batching_{batching}_ind_{target_class}_repetition_{repetition}.jpg"
                plot_tcav_scores(experimental_set, tcav_scores, filename, pathname) #todo debug plotting takes too long for debug
                print(f"plotted {filename}")

                all_tcav_scores.append(tcav_scores)

            picklefilename = f"{name}_TCAV_batching_{batching}_ind_{target_class}"
            pickle_path = os.path.join(picklefolder, f'{picklefilename}.pkl')
            with open(pickle_path, 'wb') as f:
                pickle.dump(all_tcav_scores, f)
            print(f'pickled {picklefilename}')

    return

def group_pickle_files(directory):
    # Dictionary to store grouped files
    grouped_files = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Compile regular expressions for efficiency
    class_regex = re.compile(r'ind_(\d+)')
    batching_regex = re.compile(r'batching_(True|False)')

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.pkl'):
            # Extract information from filename
            class_match = class_regex.search(filename)
            batching_match = batching_regex.search(filename)

            if class_match and batching_match:
                class_id = class_match.group(1)
                class_id = int(class_id)
                batching = batching_match.group(1)
                batching = f'batching_{batching}'

                # Determine the category of the file
                if 'absolute' in filename:
                    grouped_files['absolute'][batching][class_id]=filename
                elif 'relative' in filename:
                    grouped_files['relative'][batching][class_id]=filename

    return grouped_files

def get_dict_of_stats(experimental_set_rand,layers,all_tcav_scores,classlist):
    # Use defaultdict to avoid manual initialization of dictionaries
    dict_of_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    first_score_size = None
    score_type = "sign_count"
    for subset_idx, subset in enumerate(experimental_set_rand):
        for score_layer in layers:
            for idx in range(2):
                score_list = []
                for tcav_score in all_tcav_scores:

                    score = tcav_score["-".join([str(c.id) for c in subset])][score_layer][score_type][
                        idx].cpu()

                    # Check size consistency
                    if first_score_size is None:
                        first_score_size = score.size()
                    elif score.size() != first_score_size:
                        raise ValueError(f"Inconsistent size found: {score.size()} != {first_score_size}")

                    score_list.append(score)

                num_elements = len(score_list)

                means = np.mean(score_list, axis=0)
                std = np.std(score_list, axis=0)

                dict_of_stats[subset_idx][score_layer][score_type][idx] = {"means": means, "std": std}
                vals = [val.item() for val in score_list]
                dict_of_stats[subset_idx][score_layer][score_type][idx]["vals"] = vals

    return dict_of_stats, num_elements


def plot_abs(dict_stats, experimental_sets, pathname, num_elements, filename,layers,class_id,batching_Flag):


    fig, axs = plt.subplots(1, 2, figsize=(20, 7))  # Two subplots side-by-side

    barWidth = 0.15
    spacing = 0.2  # Spacing between bars for different concepts
    layer_positions = np.arange(len(layers))
    score_type="sign_count"

    for idx_es, concepts in enumerate(experimental_sets):

        ax = axs[idx_es]  # Select the corresponding subplot axis

        for i in range(len(concepts)):

            # Prepare positions for the bars of the current concept
            adjusted_pos = layer_positions + i * barWidth  # Shift position for each concept

            # Extract mean and std for this concept
            means = []
            stds = []
            for layer in layers:
                # Extract mean and std for the current layer
                layer_stats = dict_stats.get(idx_es, {}).get(layer, {}).get(score_type, {}).get(i, {})
                means.append(layer_stats.get('means', 0))
                stds.append(layer_stats.get('std', 0))

            # Plot bars for each concept
            bars=ax.bar(adjusted_pos, means, width=barWidth, yerr=stds, capsize=5,
                   edgecolor='white', label=f'{concepts[i].name}')

            # Add value labels with mean and +/- std on top of the bars
            for bar, mean, std in zip(bars, means, stds):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{mean:.2f} (± {std:.2f})',
                        ha='center', va='bottom', fontsize=10, color='black')

        # Setting plot details
        if batching_Flag=='batching_True': batching_Flag=True #todo ugly
        elif batching_Flag=='batching_False': batching_Flag=False
        ax.set_title(f'TCAV Scores for Subset {idx_es} - repetitions: {num_elements} - class: {class_id} - batching: {batching_Flag}', fontsize=16)
        ax.set_ylabel('Mean TCAV Score', fontsize=12)
        ax.set_xticks(layer_positions + (len(concepts) - 1) * (barWidth + spacing) / 2)
        ax.set_xticklabels(layers, fontsize=10)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=len(concepts), fontsize=10)

    # Synchronize y-axis across all subplots
    for ax in axs:
        ax.set_ylim([0, 1])

    # Adjust layout to fit the title
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the figure
    fullpath = os.path.join(pathname, f'{filename}.png')
    plt.savefig(fullpath)
    plt.close()

    return

def pickle_to_cpu(picklefiles,picklefolder):
    fullname=os.path.join(picklefolder,picklefiles)
    with open(fullname, 'rb') as f:
        all_tcav_scores = pickle.load(f)
        for tcav_scores in all_tcav_scores:
            for concept_key, concepts in tcav_scores.items():  # because cpu-conversion was not part of the pipeline from the beginning
                for layer_key, layers in concepts.items():
                    for modus_key, modus in layers.items():
                        layers[modus_key] = modus.cpu()
            torch.cuda.empty_cache()
    return all_tcav_scores

def plot_q_q(all_dicts, experimental_sets, pathname, num_elements, filename, layers, class_id, batching_Flag,combinations):

    num_sets = len(experimental_sets)
    concepts = [0, 1]  # Assuming two concepts
    score_type="sign_count"

    # Total grid size based on batching (2), classes (variable), and experimental sets (num_sets)
    num_layer = len(layers)  # Total number of classes
    fig, axs = plt.subplots(num_sets*2,num_layer, figsize=(15, 15), sharex=True, sharey=True)

    for combination, one_dict in zip(combinations,all_dicts):
        experiment_pair, batching_Flag, class_id = combination
        name_flag, experimental_set = experiment_pair

        if "relative" in name_flag:
            continue
        for idx_es, subset in enumerate(experimental_sets):  # Horizontal: experimental sets

            # Plot Q-Q plot for each concept (on the same axis)
            for concept_idx,concept in enumerate(subset):
                # Get values from the dictionary for the current experimental set, batching, and class
                row_idx = idx_es * len(subset) + concept_idx

                for idx_layer, layer in enumerate(layers):
                    column_idx=idx_layer
                    ax = axs[row_idx, column_idx]  # Select correct subplot

                    # Clear the axis before plotting to prevent multiple rows of points
                    ax.cla()

                    layer_stats = one_dict.get(idx_es, {}).get(layer, {}).get(score_type, {}).get(concept_idx, {})
                    vals=layer_stats["vals"]

                    if vals is None:
                        print("vals empty")
                    if len(vals)==0:
                        print("vals of 0 lenght")

                    # Plot the Q-Q plot for each concept
                    vals=np.array(vals)
                    #bins=len(vals)
                    bins=math.ceil(math.sqrt(len(vals)))
                    ax.hist(vals, bins=bins, alpha=0.7,range=[0,1])#,density=True)
                    #sm.qqplot(vals, line='45',ax=ax)


                    # Set axis labels only for bottom-left plots for clarity
                    if row_idx == num_layer - 1:
                        ax.set_xlabel(f'Layer {layer}')
                    if column_idx == 0:
                        ax.set_ylabel(f'Subset {idx_es}, Concept {concept_idx}')


            # Set overall figure title and adjust layout
        fig.suptitle(f'Histogram for class {class_id} batch {batching_Flag} ', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save the figure
        fullpath = os.path.join(pathname, f'histogram for class {class_id} batch {batching_Flag}.png')
        fig.savefig(fullpath)
        plt.close()

    return



if __name__ == "__main__":

    # initialize path
    concepts_path = "data/tcav/image/concepts/"

    # assemble concepts
    stripes_concept = assemble_concept("striped", 0, concepts_path=concepts_path)
    zigzagged_concept = assemble_concept("zigzagged", 1, concepts_path=concepts_path)
    dotted_concept = assemble_concept("dotted", 2, concepts_path=concepts_path)

    # assemble random concepts
    random_0_concept = assemble_concept("random_0", 3, concepts_path=concepts_path)
    random_1_concept = assemble_concept("random_1", 4, concepts_path=concepts_path)

    # load model
    model = torchvision.models.googlenet(pretrained=True)
    model = model.eval()

    model_size = get_model_size(model)
    print(f"Model size: {model_size:.2f} MB")

    #define layers
    layers = ['inception4c', 'inception4d', 'inception4e']

    #define experimental sets
    experimental_set_rand = [[stripes_concept, random_0_concept], [stripes_concept, random_1_concept]]
    experimental_set_zig_dot = [[stripes_concept, zigzagged_concept, dotted_concept]]

    # Load sample images from folder
    zebra_imgs = load_image_tensors('zebra', transform=False)

    # Load sample images from folder
    zebra_tensors = torch.stack([transform(img) for img in zebra_imgs])

    #define imagenet classes (added hippo)
    zebra_ind = 340
    hippo_ind = 344
    index_list=[zebra_ind,hippo_ind]

    # create folder structure
    pathname='./zebra-figures'
    if not os.path.exists(pathname):
        os.makedirs(pathname)

    # create folder structure
    picklefolder = './zebra-pickle'
    if not os.path.exists(pathname):
        os.makedirs(pathname)

    # define parameters
    repetition_nr=50
    batch_size=4


    #pristine TCAV

    mytcav = TCAV(model=model,
                  layers=layers,
                  layer_attr_method=LayerIntegratedGradients(
                      model, None, multiply_by_inputs=False))
    tcav_scores_w_random = mytcav.interpret(inputs=zebra_tensors,
                                            experimental_sets=experimental_set_rand,
                                            target=zebra_ind,
                                            n_steps=5,
                                            )
    filename="pristine_tcav_plot"
    pathname=pathname
    print(f'tcav_scores_w_random: {tcav_scores_w_random}')
    plot_tcav_scores(experimental_set_rand, tcav_scores_w_random, filename, pathname)

    # calculate and plot individual TCAVs
    debug=True  #todo deactivated for debug, because takes long
    if debug==False:
        calculate_tcav("absolute",experimental_set_rand,picklefolder,pathname,repetition_nr,index_list,batch_size)
        calculate_tcav("relative", experimental_set_zig_dot,picklefolder,pathname,repetition_nr,index_list,batch_size) #todo deactivated for debug

    # group pickles by absolute, relative, batching=True, batching=False
    grouped_files = group_pickle_files(picklefolder)

    # Define your sets of parameters
    experiment_pair = [('absolute',experimental_set_rand), ('relative',experimental_set_zig_dot)]
    batching_Flag = ['batching_True', 'batching_False']
    combinations = list(itertools.product(experiment_pair, batching_Flag,index_list))

    all_dicts=[]
    dict_of_mean = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))))
    for experiment_pair, batching_Flag,class_id in combinations:
        name_flag, experimental_set = experiment_pair
        filename = f'{name_flag}_{batching_Flag}_class_{class_id}'
        subset = grouped_files[name_flag][batching_Flag][class_id]
        all_tcav_scores = pickle_to_cpu(subset,picklefolder)
        dict_of_stats, num_elements = get_dict_of_stats(experimental_set, layers, all_tcav_scores, index_list)
        plot_abs(dict_of_stats, experimental_set, pathname, num_elements, filename,layers,class_id,batching_Flag)
        all_dicts.append(dict_of_stats)


        score_type="sign_count"
        for idx_es, concepts in enumerate(experimental_set):
            for i in range(len(concepts)):
                # Extract mean and std for this concept
                means = []
                stds = []
                for layer in layers:
                    # Extract mean and std for the current layer
                    layer_stats = dict_of_stats.get(idx_es, {}).get(layer, {}).get(score_type, {}).get(i, {})
                    current_mean=(layer_stats.get('means', 0))
                    current_std=(layer_stats.get('std', 0))
                    dict_of_mean[name_flag][concepts[i].name][layer][class_id][batching_Flag] = {"mean": current_mean, "std": current_std}


    for layer in layers:
        for cname in ["striped","random_0","random_1"]:
            class_id=zebra_ind
            mean1=dict_of_mean["absolute"][cname][layer][class_id]['batching_True']['mean']
            std1=dict_of_mean["absolute"][cname][layer][class_id]['batching_True']['std']
            n1=num_elements
            mean2=dict_of_mean["absolute"][cname][layer][class_id]['batching_False']['mean']
            std2=dict_of_mean["absolute"][cname][layer][class_id]['batching_False']['std']
            n2=num_elements
            _, p_value = ttest_ind_from_stats(mean1,std1,n1,mean2,std2,n2, equal_var=True, alternative='two-sided')
            print(f'layer: {layer}')
            print(f'concept: {cname}')
            print(f"P-value: {p_value}")
            alpha=0.05
            if p_value < alpha:
                print("statistically significant")
            else:
                print("statistically insignificant")

    plot_q_q(all_dicts, experimental_set_rand, pathname, num_elements, filename, layers, class_id, batching_Flag,combinations)





