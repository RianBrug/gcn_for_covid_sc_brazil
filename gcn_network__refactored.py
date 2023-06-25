import sqlite3
import sklearn
import torch
from torch import nn
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.nn import Linear, GRU, Dropout, functional as F
from torch.optim import Adam
import torch.multiprocessing as mp
from sklearn.model_selection import TimeSeriesSplit
from utils import Utils
import optuna
from optuna import Trial
from optuna.pruners import SuccessiveHalvingPruner
import pandas as pd
import numpy as np
import json
from collections import defaultdict
import os
from pathlib import Path
from networkx import to_undirected
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_FEATURES = 1
WINDOW_SIZE = 15
TRAINING_EPOCHS = 150
TOTAL_EPOCHS = 160
BATCH_SIZE = 5
NUM_SPLITS = 3
DATABASE_URL = 'sqlite:///gcn_new'
N_TRAILS = 100
STUDY_NAME = 'gcn_new'
HOME_DIR = Path("/Users/rcvb/Documents/tcc_rian/code")
N_JOBS = 5

class TemporalGNN(nn.Module):
    def __init__(self, NUM_FEATURES, num_hidden_channels, dropout):
        super(TemporalGNN, self).__init__()
        self.conv1 = GCNConv(NUM_FEATURES, num_hidden_channels)
        self.gru = GRU(WINDOW_SIZE, WINDOW_SIZE)
        self.lin = Linear(num_hidden_channels, 1)  # output one value
        self.dropout = Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        seq_len, num_features_from_shape = x.shape
        assert num_features_from_shape == self.conv1.in_channels, "Expected input tensor dimension does not match with num_features"
        
        x = self.dropout(F.relu(self.conv1(x, edge_index)))
        
        # reshape x for GRU input
        num_nodes = x.shape[0] // self.conv1.in_channels
        x = x.t()
        x, _ = self.gru(x)
        x = x.transpose(0, 1).contiguous().view(seq_len, -1)  # reshape x back to original form
        
        x = self.lin(x)
        return x


def load_and_preprocess_data():
    with open(HOME_DIR / 'assets/confirmed_cases_by_region_and_date.json') as file:
        data = json.load(file)

    df = pd.DataFrame(data)
    df.reset_index(inplace=True)
    df.rename(columns={'index':'collect_date'}, inplace=True)
    df['collect_date'] = pd.to_datetime(df['collect_date'])
    df.sort_values(by=['collect_date'], inplace=True)
    df.fillna(0, inplace=True)

    return df

def create_graphs(df):
    graphs = {}
    # Exclude the collect_date column
    for region in df.columns:
        if region != 'collect_date':
            graphs[region] = defaultdict(list)

    for _, row in df.iterrows():
        for region, confirmed_cases in row.iteritems():
            if region != 'collect_date':
                graphs[region]["cases"].append(confirmed_cases)
    
    # create column 'index' to be used in edge_index
    for i, region in enumerate(graphs.keys()):
        graphs[region]["index"] = i

    return graphs

def add_neighbors(graphs):
    for region in graphs.keys():
        neighbors = Utils.get_neighbors_of_region(region)
        graphs[region]["neighbors"] = neighbors

    return graphs

def graphs_to_data(graphs, populations, sequence_length=15):
    data_list = []
    max_population = max(populations.values())
    for region, graph in graphs.items():
        confirmed_cases = torch.tensor(graph["cases"], dtype=torch.float)
        neighbors_indices = torch.tensor([graphs[n]["index"] for n in graph["neighbors"]], dtype=torch.long)
        neighbors_populations = torch.tensor([populations[n] for n in graph["neighbors"]], dtype=torch.float) / max_population
        edge_index = torch.stack([torch.tensor([graphs[region]['index']]*len(neighbors_indices), dtype=torch.long), neighbors_indices], dim=0)
        edge_attr = neighbors_populations.view(-1, 1)

        for i in range(len(confirmed_cases) - sequence_length):
            x = confirmed_cases[i:i+sequence_length].view(-1, 1)  # reshape to be two-dimensional
            y = confirmed_cases[i:i+sequence_length+1]
            data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, y_window=sequence_length)
            data_list.append(data)

    return data_list


def split_data(data_list, num_splits):
    tscv = TimeSeriesSplit(n_splits=num_splits)
    split_data = []
    for train_index, test_index in tscv.split(data_list):
        split_data.append((train_index, test_index))
    return split_data

def train_and_evaluate(data_list, split_data, trial):
    num_hidden_channels = trial.suggest_categorical("num_hidden_channels", [15, 30, 45, 90, 180])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    model = TemporalGNN(NUM_FEATURES, num_hidden_channels, dropout).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()

    for epoch in range(TOTAL_EPOCHS):
        # Training
        for train_index, _ in split_data:
            model.train()
            for index in train_index:
                data = data_list[index].to(DEVICE)
                optimizer.zero_grad()
                out = model(data)
                loss = criterion(out[data.y_window:], data.y[data.y_window:])
                loss.backward()
                optimizer.step()

        # Evaluation
        if epoch >= TRAINING_EPOCHS:
            for _, test_index in split_data:
                model.eval()
                for index in test_index:
                    data = data_list[index].to(DEVICE)
                    out = model(data)
                    loss = criterion(out[data.y[:len(test_index)]], data.y[:len(test_index)])
                    trial.report(loss, epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

    return loss.item()

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {"MAE": mae, "R2": r2, "MSE": mse, "MAPE": mape}

def plot_predictions(y_true, y_pred, title):
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label="True")
    plt.plot(y_pred, label="Predicted")
    plt.title(title)
    plt.legend()
    plt.show()

def evaluate_model(model, data_list, test_indices):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for index in test_indices:
            data = data_list[index].to(DEVICE)
            out = model(data)
            y_true.append(data.y[data.y_window:].cpu().numpy())
            y_pred.append(out[data.y_window:].cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return y_true, y_pred

def main():
    print("------- VERSIONS -------")
    print("SQLite version: ", sqlite3.version)
    print("Optuna version: ", optuna.__version__)
    print("PyTorch version: ", torch.__version__)
    print("Pandas version: ", pd.__version__)
    print("Numpy version: ", np.__version__)
    print("Sklearn version: ", sklearn.__version__)
    print("Torch Geometric version: ", torch_geometric.__version__)
    print("-------------------------------------")

    # Enable parallel processing
    # mp.set_start_method('spawn')
    df = load_and_preprocess_data()
    graphs = create_graphs(df)
    graphs = add_neighbors(graphs)
    populations = Utils.get_population_from_csv()

    data_list = graphs_to_data(graphs, populations)
    data_split = split_data(data_list, NUM_SPLITS)

    pruner = SuccessiveHalvingPruner()
    study = optuna.create_study(
        direction='minimize', pruner=pruner, study_name=STUDY_NAME, 
        storage=DATABASE_URL, load_if_exists=True)
    study.optimize(lambda trial: train_and_evaluate(data_list, data_split, trial), n_trials=N_TRAILS, 
                   n_jobs=N_JOBS, show_progress_bar=True)

    print("Best trial:")
    trial = study.best_trial
    print(" Value: ", trial.value)
    print(" Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Train final model with best parameters from the Optuna study
    num_hidden_channels = trial.params["num_hidden_channels"]
    dropout = trial.params["dropout"]
    lr = trial.params["lr"]
    model = TemporalGNN(NUM_FEATURES, num_hidden_channels, dropout).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()

    for epoch in range(TOTAL_EPOCHS):
        for train_index, _ in data_split:
            model.train()
            for index in train_index:
                data = data_list[index].to(DEVICE)
                optimizer.zero_grad()
                out = model(data)
                loss = criterion(out[data.y_window:], data.y[data.y_window:])
                loss.backward()
                optimizer.step()

    for _, test_index in data_split:
        y_true, y_pred = evaluate_model(model, data_list, test_index)
        metrics = calculate_metrics(y_true, y_pred)
        print(f"Metrics: {metrics}")
        plot_predictions(y_true, y_pred, "True vs Predicted Confirmed Cases")

if __name__ == "__main__":
    main()
