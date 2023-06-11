import json
from matplotlib import pyplot as plt
import sklearn
import torch
import networkx as nx
import pandas as pd
import torch_geometric
from torch_geometric.utils import from_networkx
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.nn import GCNConv
import numpy as np
import optuna
import sqlite3
import torch.multiprocessing as mp

print("-------------------------------------")
print("SQLite version: ", sqlite3.version)
print("Optuna version: ", optuna.__version__)
print("PyTorch version: ", torch.__version__)
print("NetworkX version: ", nx.__version__)
print("Pandas version: ", pd.__version__)
print("Numpy version: ", np.__version__)
print("Sklearn version: ", sklearn.__version__)
print("Torch Geometric version: ", torch_geometric.__version__)
print("-------------------------------------")

# Enable parallel processing
mp.set_start_method('spawn')

# Load and preprocess data
home = '/Users/rcvb/Documents/tcc_rian/code'

# Load and preprocess data
with open(f'{home}/assets/confirmed_cases_by_region_and_date.json') as file:
    data = json.load(file)

df = pd.DataFrame(data)
df.reset_index(inplace=True)
df.rename(columns={'index':'collect_date'}, inplace=True)
df['collect_date'] = pd.to_datetime(df['collect_date'])
df.sort_values(by=['collect_date'], inplace=True)
df.drop(df.columns[len(df.columns)-1], axis=1, inplace=True)

# print total number of rows
print("Total number of rows: ", df.shape[0])

outliers = {}
# Iterate over each region
for region in df.columns[1:]:
    # Calculate IQR
    Q1 = df[region].quantile(0.25)
    Q3 = df[region].quantile(0.75)
    IQR = Q3 - Q1

    factor = 3.8
    # Define outliers
    outlier_condition = (df[region] < (Q1 - factor * IQR)) | (df[region] > (Q3 + factor * IQR))

    # Save outlier data in dictionary
    outliers[region] = df[outlier_condition]

    # Remove outliers
    df = df[~outlier_condition]

print("Outliers:")
for region, outlier_data in outliers.items():
    print(f"\nRegion: {region}")
    print(outlier_data)

outlier_counts = {region: outlier_data.shape[0] for region, outlier_data in outliers.items()}
total_outliers = sum(outlier_counts.values())

print("Outlier counts:")
for region, count in outlier_counts.items():
    print(f"\nRegion: {region}")
    print("Number of outliers: ", count)

print("Total number of outliers: ", total_outliers)

