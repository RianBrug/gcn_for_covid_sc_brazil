import json
import sklearn
from sklearn import datasets
import torch
import networkx as nx
import pandas as pd
import torch_geometric
from torch_geometric.utils import from_networkx
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from torch_geometric.nn import GCNConv
from torch.nn import L1Loss, MSELoss
import torch.nn.functional as F
from torch.optim import Adam, RMSprop
import torch.nn as nn

from utils import Utils
import numpy as np
import optuna
import sqlite3
import optuna.visualization as vis
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.multiprocessing as mp
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from optuna.pruners import SuccessiveHalvingPruner
from sqlalchemy import create_engine

# engine = create_engine('postgresql://rcvb:@localhost:5432/gnns_db')
# database_url = 'postgresql://rcvb:@localhost:5432/gnns_db'

print("------- VERSIONS -------")
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
with open(f'{home}/assets/confirmed_cases_by_region_and_date.json') as file:
    data = json.load(file)

df = pd.DataFrame(data)
df.reset_index(inplace=True)
df.rename(columns={'index':'collect_date'}, inplace=True)
df['collect_date'] = pd.to_datetime(df['collect_date'])
df.sort_values(by=['collect_date'], inplace=True)
# df.drop(df.columns[len(df.columns)-1], axis=1, inplace=True)
df.fillna(0, inplace=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
database_url = 'sqlite:///gcn_latest'
window = 15
total_epochs = 150
trials_until_start_pruning = 150
n_trails = 10
n_jobs = 5 # Number of parallel jobs
num_original_features = window  # original size
num_additional_features = 3  # new additional features
patience_learning_scheduler = 130
true_values = []
predictions = []

tscv = TimeSeriesSplit(n_splits=3)

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, 1)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_channels, nhead=8),
            num_layers=3
        )
        
        # Define positional embeddings
        self.position_embedding = nn.Embedding(10000, 18)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Add positional embeddings to input
        # positions = torch.arange(0, x.size(1)).expand(x.size(0), -1).to(x.device)
        positions = torch.arange(x.size(1)).unsqueeze(0).repeat(x.size(0), 1).to(x.device)
        x = x + self.position_embedding(positions)

        # Apply transformer to sequence of feature vectors for each node
        x = self.transformer_encoder(x)

        # Use all feature vectors of the sequence as input to the graph convolutional layers
        x = x.view(-1, x.size(-1))

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)

        return x
    
    def compute_loss(self, pred, data):
        # Use Mean Absolute Error (MAE) as the loss
        return F.l1_loss(pred, data.y.view(-1))

def sliding_windows(data, window):
        X = []
        Y = []

        for i in range(len(data)-2*window):
            X.append(data.iloc[i:i+window].values)
            Y.append(data.iloc[i+window:i+2*window].values)

        return np.array(X), np.array(Y)

def data_to_graph(df, window, train_indices, val_indices):
    G = nx.Graph()
    train_mask = []
    val_mask = []

    # Load your additional features data here
    pr_df = pd.read_csv(f'{home}/assets/populacao_residente_sc_por_macroregiao.csv', sep=";", index_col=0)
    rf_df = pd.read_csv(f'{home}/assets/recursos_fisicos_hospitalares_leitos_de_internação_por_macro_out22.csv', sep=";", index_col=0)
    aa_df = pd.read_csv(f'{home}/assets/abastecimento_agua_por_populacao.csv', sep=";", index_col=0)

    for region in df.columns[1:]:
        region_df = df[['collect_date', region]].fillna(0)
        X, Y = sliding_windows(region_df[region], window)

        # Retrieve additional features for the current region
        add_features = np.array([
            pr_df.loc[region],
            rf_df.loc[region],
            aa_df.loc[region]
        ]).flatten()

        # Create the node for the region and add the feature vectors and labels for each time window
        features = []
        labels = []
        for i in range(len(X)):
            # Concatenate original features with additional features
            current_features = np.concatenate([X[i], add_features]).astype(np.float32)
            features.append(current_features)
            labels.append(Y[i])
        
        G.add_node(region, x=torch.stack([torch.tensor(f) for f in features]), y=torch.tensor(labels).float())

        # Add edges to neighboring regions
        for neighbor in Utils.get_neighbors_of_region(region):
            if neighbor in G.nodes:
                G.add_edge(region, neighbor)
        
        # Masking for train and validation datasets
        # Now each region is a single node, so we can use the train_indices and val_indices directly
        if region in train_indices:
            train_mask.append(True)
        else:
            train_mask.append(False)
        if region in val_indices:
            val_mask.append(True)
        else:
            val_mask.append(False)

    data = from_networkx(G)
    data.train_mask = torch.tensor(train_mask)
    data.val_mask = torch.tensor(val_mask)

    return data


def train(model, data, train_mask, optimizer):
    model.train()
    optimizer.zero_grad()
    pred = model(data)[train_mask]
    loss = model.objective(pred, data)
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, data, test_mask):
    model.eval()
    with torch.no_grad():
        pred = model(data)[test_mask]
    return pred, data.y[test_mask]

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    hidden_channels = trial.suggest_categorical('hidden_channels', [16, 32, 64, 128])
    dropout = trial.suggest_float('dropout', 0.1, 0.7)
    model = GCN(hidden_channels=hidden_channels)
    optimizer = Adam(model.parameters(), lr=lr)

    for i in range(total_epochs):
        for train_indices, val_indices in tscv.split(df):
            data = data_to_graph(df, window, train_indices, val_indices)

            loss = train(model, data, data.train_mask, optimizer)
            trial.report(loss, i)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    # Validate model
    for train_indices, val_indices in tscv.split(df):
        data = data_to_graph(df, window, train_indices, val_indices)

        pred, y = test(model, data, data.val_mask)
        score = mean_absolute_error(pred.cpu(), y.cpu())
        trial.report(score, total_epochs)

    return score

study = optuna.create_study(direction="minimize", pruner=SuccessiveHalvingPruner(min_resource=1, reduction_factor=4, min_early_stopping_rate=0))
study.optimize(objective, n_trials=n_trails, timeout=600, n_jobs=n_jobs)
print(study.best_trial)
