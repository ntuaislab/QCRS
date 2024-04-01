import numpy as np
import matplotlib

# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from typing import *
import pandas as pd
import seaborn as sns
import math
from dataclasses import dataclass, field
from ipdb import set_trace as st
import os

'''
analysis 0703  
'''

print("Ploting...")


class Accuracy(object):
    def at_radii(self, radii: np.ndarray):
        raise NotImplementedError()


@dataclass()
class ApproximateAccuracy(Accuracy):
    data_file_path: str
    def __post_init__(self):
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        self.acr = (df["correct"] * df["radius"]).mean()


    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        return np.array([self.at_radius(df, radius) for radius in radii])

    def at_radius(self, df: pd.DataFrame, radius: float):
        return (df["correct"] & (df["radius"] >= radius)).mean()


@dataclass()
class HighProbAccuracy(Accuracy):
    data_file_path: str 
    alpha: float 
    rho: float

    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        return np.array([self.at_radius(df, radius) for radius in radii])

    def at_radius(self, df: pd.DataFrame, radius: float):
        mean = (df["correct"] & (df["radius"] >= radius)).mean()
        num_examples = len(df)
        return (mean - self.alpha - math.sqrt(self.alpha * (1 - self.alpha) * math.log(1 / self.rho) / num_examples)
                - math.log(1 / self.rho) / (3 * num_examples))


class Line(object):
    def __init__(self, quantity: Accuracy, legend: str, plot_fmt: str = "", scale_x: float = 1, color='r', width=1.5):
        self.quantity = quantity
        self.legend = legend + f':{quantity.acr:.3f}'
        self.plot_fmt = plot_fmt
        self.scale_x = scale_x
        self.color = color
        self.width = width


def plot_certified_accuracy(outfile: str, title: str, max_radius: float,
                            lines: List[Line], radius_step: float = 0.01) -> None:
    radii = np.arange(0, max_radius + radius_step, radius_step)
    plt.figure(figsize=(6,5))
    for line in lines:
        plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), line.plot_fmt, linewidth=line.width, color=line.color)

    plt.ylim((0, 1))
    plt.xlim((0, max_radius))
    plt.tick_params(labelsize=23)
    plt.xlabel("radius", fontsize=25)
    # plt.ylabel("certified accuracy", fontsize=25)
    plt.legend([method.legend for method in lines], loc='upper right', fontsize=25,framealpha=0.5)
    plt.tight_layout()
    # plt.title(title, fontsize=23)
    plt.tight_layout(pad=0)
    
    plt.savefig(outfile + ".pdf", bbox_inches='tight', pad_inches=0.01)
    plt.savefig(outfile + ".png", dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.close()


def smallplot_certified_accuracy(outfile: str, title: str, max_radius: float,
                                 methods: List[Line], radius_step: float = 0.01, xticks=0.5) -> None:
    radii = np.arange(0, max_radius + radius_step, radius_step)
    plt.figure()
    for method in methods:
        plt.plot(radii, method.quantity.at_radii(radii), method.plot_fmt)

    plt.ylim((0, 1))
    plt.xlim((0, max_radius))
    plt.xlabel("radius", fontsize=22)
    plt.ylabel("certified accuracy", fontsize=22)
    plt.tick_params(labelsize=20)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(xticks))
    plt.legend([method.legend for method in methods], loc='upper right', fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile + ".pdf")
    plt.close()


def latex_table_certified_accuracy(outfile: str, radius_start: float, radius_stop: float, radius_step: float,
                                   methods: List[Line]):
    radii = np.arange(radius_start, radius_stop + radius_step, radius_step)
    accuracies = np.zeros((len(methods), len(radii)))
    for i, method in enumerate(methods):
        accuracies[i, :] = method.quantity.at_radii(radii)

    f = open(outfile, 'w')

    for radius in radii:
        f.write("& $r = {:.3}$".format(radius))
    f.write("\\\\\n")

    f.write("\midrule\n")

    for i, method in enumerate(methods):
        f.write(method.legend)
        for j, radius in enumerate(radii):
            if i == accuracies[:, j].argmax():
                txt = r" & \textbf{" + "{:.2f}".format(accuracies[i, j]) + "}"
            else:
                txt = " & {:.2f}".format(accuracies[i, j])
            f.write(txt)
        f.write("\\\\\n")
    f.close()


def markdown_table_certified_accuracy(outfile: str, radius_start: float, radius_stop: float, radius_step: float,
                                      methods: List[Line]):
    radii = np.arange(radius_start, radius_stop + radius_step, radius_step)
    accuracies = np.zeros((len(methods), len(radii)))
    for i, method in enumerate(methods):
        accuracies[i, :] = method.quantity.at_radii(radii)

    f = open(outfile, 'w')
    f.write("|  | ")
    for radius in radii:
        f.write("r = {:.3} |".format(radius))
    f.write("\n")

    f.write("| --- | ")
    for i in range(len(radii)):
        f.write(" --- |")
    f.write("\n")

    for i, method in enumerate(methods):
        f.write("<b> {} </b>| ".format(method.legend))
        for j, radius in enumerate(radii):
            if i == accuracies[:, j].argmax():
                txt = "{:.2f}<b>*</b> |".format(accuracies[i, j])
            else:
                txt = "{:.2f} |".format(accuracies[i, j])
            f.write(txt)
        f.write("\n")
    f.close()

print('generating')
if __name__ == "__main__":
    target_path = 'analysis/my_plots_final6'
    # os.mkdir(target_path)
    os.makedirs(target_path, exist_ok=True)

    plot_certified_accuracy(
            f"{target_path}/cifar_qcrs", "CIFAR10, $\sigma=.50$", 2.8, [
                Line(ApproximateAccuracy("exp/cifar10/cifar_exp"), "qcrs", plot_fmt='-', color='r', width=1.6),
            ])
    plot_certified_accuracy(
            f"{target_path}/imagenet_qcrs", "ImageNet, $\sigma=.50$", 2.8, [
                Line(ApproximateAccuracy("exp/imagenet/imagenet_exp"), "qcrs", plot_fmt='-', color='r', width=1.6),
            ])
    print(f'Save at {target_path}')
        
