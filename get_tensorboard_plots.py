from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import tensorflow as tf

filename='tensorboardplot_model_2'

roc_auc_value='roc_auc_value'
validation_loss='validation loss'
training_loss='training loss'

# Path to your TensorBoard log directory or a specific event file
log_dirs =['/home/fkraehenbuehl/projects/CaptumTCAV/prep-model/tensorboard_results/tensorboard_logs_wo_concepts_bilinear_lr0.0001_epoches5/events.out.tfevents.1727356320.nerve.12555.0',
'/home/fkraehenbuehl/projects/CaptumTCAV/prep-model/tensorboard_results/tensorboard_logs_w_concepts_bilinear_lr0.0001_epoches5/events.out.tfevents.1727467350.nerve.9788.0',
'/home/fkraehenbuehl/projects/CaptumTCAV/prep-model/tensorboard_results/tensorboard_logs_w_marker_bilinear_lr0.0001_epoches5/events.out.tfevents.1727422746.nerve.29071.0']


def get_scalar_run_tensorboard(tag, filepath):
    values, steps = [], []
    for e in tf.compat.v1.train.summary_iterator(filepath):
        if len(e.summary.value) > 0:  # Skip first empty element
            if e.summary.value[0].tag == tag:
                tensor = e.summary.value[0].tensor
                value, step = (
                    tf.io.decode_raw(tensor.tensor_content, tf.float32)[0].numpy(),
                    e.step
                )
                values.append(value)
                steps.append(step)
            else:
                print(f"No values found for tensor at step: {e.step}")
        else:
            print(f"Empty tensor for tag: {tag} at step: {e.step}")
    return values, steps

# Exponential Moving Average Smoothing
def smooth_curve(values, alpha=0.99):
    smoothed_values = []
    for i, val in enumerate(values):
        if i == 0:
            smoothed_values.append(val)  # No smoothing for the first value
        else:
            smoothed_values.append(alpha * val + (1 - alpha) * smoothed_values[-1])
    return smoothed_values

metrics=["roc_auc_val","validation loss","training loss"]
colors = ['blue', 'orange', 'green']  # Colors for different models

for metric in metrics:

    for log_dir, color in zip(log_dirs, colors):

        values, steps = get_scalar_run_tensorboard(metric, log_dir)

        plt.plot(steps, values, label=f'Model {log_dirs.index(log_dir) + 1}', color=color)

    # Set labels and title
    plt.xlabel('Steps')
    plt.ylabel('Smoothed Values')
    plt.title(metric.replace("_", ""))
    plt.legend()
    plt.grid()

    # Save the plot for the current metric
    plt.savefig(f'{metric.replace(" ", "_").lower()}.png')
    plt.close()
