import pickle
import os
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import random as rd
from matplotlib import pylab
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

folder = Path('./project4/cifar-10-batches-py')
for file in folder.iterdir():
     if file.name.endswith("1"):
        data1 = unpickle(file)