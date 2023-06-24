import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.nn import Linear, GRU, Dropout, functional as F
from torch.optim import Adam
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

# Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WINDOW_SIZE = 15
TRAINING_EPOCHS = 150
TOTAL_EPOCHS = 160
BATCH_SIZE = 5
NUM_SPLITS = 3
DATABASE_URL = 'sqlite:///gcn_latest'
N_TRAILS = 100
STUDY_NAME = 'gcn_refactored'
HOME_DIR = Path("/Users/rcvb/Documents/tcc_rian/code")

class TemporalGNN(nn.Module):
    def __init__(self, num_features, num_hidden_channels, dropout):
        super(TemporalGNN, self).__init__()
        self.conv1 = GCNConv(num_features, num_hidden_channels)
        self.gru = GRU(num_hidden_channels, num_hidden_channels)
        self.lin = Linear(num_hidden_channels, 1)  # output one value
        self.dropout = Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.dropout(F.relu(self.conv1(x, edge_index)))
        x, _ = self.gru(x.unsqueeze(0))
        x = x.squeeze(0)
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

    return graphs

def add_neighbors(graphs):
    for region in graphs.keys():
        neighbors = Utils.get_neighbors_of_region(region)
        graphs[region]["neighbors"] = neighbors

    return graphs

def graphs_to_data(graphs, populations):
    data_list = []
    max_population = max(populations.values())
    for region, graph in graphs.items():
        confirmed_cases = torch.tensor(graph["cases"], dtype=torch.float).view(-1, 1)
        neighbors_indices = torch.tensor([graphs[n]["index"] for n in graph["neighbors"]], dtype=torch.long)
        neighbors_populations = torch.tensor([populations[n] for n in graph["neighbors"]], dtype=torch.float) / max_population
        edge_index = torch.stack([torch.tensor([graphs[region]["index"]]*len(neighbors_indices), dtype=torch.long), neighbors_indices], dim=0)
        edge_attr = neighbors_populations.view(-1, 1)
        data = Data(x=confirmed_cases, edge_index=edge_index, edge_attr=edge_attr)
        data_list.append(data)
    return data_list


def split_data(data_list, num_splits):
    tscv = TimeSeriesSplit(n_splits=num_splits)
    split_data = []
    for train_index, test_index in tscv.split(data_list):
        split_data.append((train_index, test_index))
    return split_data

def train_and_evaluate(data_list, split_data, trial):
    num_hidden_channels = trial.suggest_categorical("num_hidden_channels", [6, 32, 48, 128, 256])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    model = TemporalGNN(WINDOW_SIZE, num_hidden_channels, dropout).to(DEVICE)
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
                loss = criterion(out[data.y[:len(train_index)]], data.y[:len(train_index)])
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

def main():
    df = load_and_preprocess_data()
    graphs = create_graphs(df)
    graphs = add_neighbors(graphs)
    populations = Utils.get_population_from_csv()    

    data_list = graphs_to_data(graphs, populations)
    data_split = split_data(data_list, NUM_SPLITS)
    pruner = SuccessiveHalvingPruner()
    study = optuna.create_study(direction='minimize', pruner=pruner, study_name=STUDY_NAME, storage=DATABASE_URL, load_if_exists=True)
    study.optimize(lambda trial: train_and_evaluate(data_list, data_split, trial), n_trials=N_TRAILS)

    print("Best trial:")
    trial = study.best_trial
    print(" Value: ", trial.value)
    print(" Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    main()
