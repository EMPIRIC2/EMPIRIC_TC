import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import seaborn_image as isns
import os
import unittest

from MachineLearning.Evaluation.site_metrics import site_squared_error
from MachineLearning.Evaluation.evaluation_utils import get_many_site_values, get_site_name
from MachineLearning.Evaluation.evaluation_testing_utils import *
from MachineLearning.Evaluation.relative_change_metrics import compute_changes_between_2_samples

def example_site_ensemble_boxplot_figure(all_site_outputs, save_path=None):
    '''
    Outputs boxplot showing the distributions of values at the first 10 sites across all the outputs
    Compares all the models over each site.

    all_site_outputs: dict of {model_name: site_outputs}
    '''

    data = []
    for i in range(10):
        for model, site_outputs in all_site_outputs.items():
            for j in range(len(site_outputs)):
                data.append({"Site Name": get_site_name(i), "Count": site_outputs[j][i], "Model": model})

    df = pd.DataFrame(data)
    sns.boxplot(data=df, x="Site Name", y="Count", hue="Model")

    if save_path is not None:
        plt.savefig(save_path)
        plt.clf()
    else:
        plt.show()

def plot_quantile_maps(ground_statistics, model_statistics, save_path=None):
    """
    Outputs images of the quantiles of both the ground outputs and the model outputs. Quantiles are 0, .25, .5, .75, 1.
    The figure has two rows of 5 images. The top row is the quantiles from ground, increasing from left to right.
    The bottom row is the quantiles from model, also increasing from left to right.

    If called with a save path, it saves the figure to that path. Otherwise, it displays the image.
    """

    ground_quantiles = ground_statistics["Quantiles"]

    model_quantiles = model_statistics["Quantiles"]
    images = np.array([ground_quantiles, model_quantiles])
    images = images.reshape((10, 110, 210))

    g = isns.ImageGrid(images, col_wrap=5, axis=0, vmin=0, vmax=16, cbar=True)

    if save_path is not None:
        plt.savefig(save_path)
        plt.clf()
    else:
        plt.show()

def ks_statistic_map(metrics, save_path = None):
    """
    Outputs a map of the KS statistic
    """
    ks = metrics["Kolmogorov-Smirnov"]
    plt.title("Kolmogorov-Smirnov Statistic of STORM vs {}".format(metrics["Model"]))
    plt.imshow(ks)
    plt.colorbar()

    if save_path is not None:
        plt.savefig(save_path)
        plt.clf()
    else:
        plt.show()
    # TODO: improve figure

def metrics_df(all_model_metrics):
    """
    Creates a dataframe for metrics. Used to save the metrics in a latex table.
    """
    for model_metrics in all_model_metrics:
        # remove this entry because it can't go into the DF
        model_metrics.pop("Kolmogorov-Smirnov")
        model_metrics.pop("Relative Error Examples")

    df = pd.DataFrame(all_model_metrics)

    return df

def top_relative_error_maps(top_error_maps, save_path=None):
    """
    Outputs maps showing the largest relative errors. Maximum of 10 images.
    """

    if len(top_error_maps) > 10: top_error_maps = top_error_maps[:10]

    g = isns.ImageGrid(top_error_maps, col_wrap=5, axis=0, vmin=0, vmax=2, cbar=True)
    if save_path is not None:
        plt.savefig(save_path)
        plt.clf()
    else: plt.show()

def save_metrics_as_latex(all_model_metrics, save_path):
    """
    Save the metrics dataframe in a latex table.
    """
    df = metrics_df(all_model_metrics)
    with open(save_path, 'w') as tf:
        tf.write(df.to_latex())

def plot_example_site_boxplot(ground_outputs, model_outputs, n_examples, save_path=None):
    """
    Outputs a boxplot showing n_examples distributions of site errors for different outputs.
    """
    box_plot_data = []
    print(len(ground_outputs), len(model_outputs))
    print("output shape")
    print(ground_outputs[0].shape)
    for i in range(n_examples):
        site_errors = site_squared_error(model_outputs[i], ground_outputs[i])
        print(site_errors.shape)
        for j in range(site_errors.shape[0]):
            print(site_errors[j].shape)

            box_plot_data.append({"Site Squared Error": site_errors[j], "Test Example": i})

    df = pd.DataFrame(box_plot_data)
    b = sns.boxplot(df, x="Test Example", y="Site Squared Error")
    b.set_xticklabels(b.get_xticks(), size = 5)
    if save_path is not None:
        plt.savefig(save_path)
        plt.clf()
    else: plt.show()

def make_figures(ground_outputs, model_outputs, ground_statistics, model_statistics, metrics, save_folder):
    ## master function to run all the figures for model Evaluation and visualization

    site_outputs = get_many_site_values(ground_outputs)
    site_predictions = get_many_site_values(model_outputs)
    
    example_site_ensemble_boxplot_figure({"STORM": site_outputs, "UNet": site_predictions}, os.path.join(save_folder, "site_ensemble_boxplot.png"))
    
    ks_statistic_map(metrics, os.path.join(save_folder, "ks_statistics.png"))
    
    plot_quantile_maps(ground_statistics, model_statistics, os.path.join(save_folder, "quantile_maps.png"))
    
    top_relative_error_maps(metrics["Relative Error Examples"], os.path.join(save_folder, "worst_relative_errors.png"))
    
    plot_example_site_boxplot(ground_outputs, model_outputs, 10, os.path.join(save_folder, "example_site_boxplots.png"))

class TestFigures(unittest.TestCase):
    """ Figure Tests """
    """ These will open example figures that must be closed for tests to complete """
    def test_ensemble_boxplot(self):
        ## Generate test data

        outputs, predictions, storm_statistics, unet_statistics, all_metrics = get_test_statistics_and_metrics()

        example_site_ensemble_boxplot_figure({"STORM": get_many_site_values(outputs), "UNet": get_many_site_values(predictions)})

    def test_example_site_error_boxplot(self):

        outputs, predictions = get_outputs_and_predictions()

        plot_example_site_boxplot(outputs, predictions, 4, "")

    def test_ks_statistics_map(self):

        outputs, predictions, storm_statistics, unet_statistics, all_metrics = get_test_statistics_and_metrics()

        ks_statistic_map(all_metrics)

    def test_quantile_maps(self):
        outputs, predictions, storm_statistics, unet_statistics, all_metrics = get_test_statistics_and_metrics()
        plot_quantile_maps(storm_statistics, unet_statistics)

    def test_relative_change_error_map(self):
        outputs, predictions = get_outputs_and_predictions()
        error_map, total_error = compute_changes_between_2_samples(outputs, predictions, 0, 1)
        top_relative_error_maps(top_error_maps=[error_map, error_map])

if __name__ == "__main__":
    unittest.main()