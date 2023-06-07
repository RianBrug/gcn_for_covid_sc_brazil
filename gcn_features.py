import json
import sklearn
import torch
import networkx as nx
import pandas as pd
import torch_geometric
from torch_geometric.utils import from_networkx
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.nn import GCNConv
from torch.nn import MSELoss
import torch.nn.functional as F
from torch.optim import Adam
from utils import Utils
import numpy as np
from sklearn.model_selection import KFold
import optuna
import sqlite3

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

# Load and preprocess data
home = '/Users/rcvb/Documents/tcc_rian/code'
with open(f'{home}/assets/confirmed_cases_by_region_and_date.json') as file:
    data = json.load(file)

df = pd.DataFrame(data)
df.reset_index(inplace=True)
df.rename(columns={'index':'collect_date'}, inplace=True)
df['collect_date'] = pd.to_datetime(df['collect_date'])
df.sort_values(by=['collect_date'], inplace=True)
df.drop(df.columns[len(df.columns)-1], axis=1, inplace=True)

window = 15
kf = KFold(n_splits=5, shuffle=True, random_state=42)

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

    # Load your additional features data here. For example:
    pr_df = pd.read_csv(f'{home}/assets/populacao_residente_sc_por_macroregiao.csv', sep=";", index_col=0)
    rf_df = pd.read_csv(f'{home}/assets/recursos_fisicos_hospitalares_leitos_de_internação_por_macro_out22.csv', sep=";", index_col=0)
    aa_df = pd.read_csv(f'{home}/assets/abastecimento_agua_por_populacao.csv', sep=";", index_col=0)

    for region in df.columns[1:]:
        region_df = df[['collect_date', region]].dropna()
        X, Y = sliding_windows(region_df[region], window)
        
        # Retrieve additional features for the current region
        add_features = np.array([
            pr_df.loc[region],
            rf_df.loc[region],
            aa_df.loc[region]
        ]).flatten()

        for i in range(len(X)):
            # Concatenate original features with additional features
            features = np.concatenate([X[i], add_features]).astype(np.float32)
            G.add_node((region, i), x=torch.tensor(features), y=torch.tensor(Y[i]).float())
            for neighbor in Utils.get_neighbors_of_region(region):
                if (neighbor, i) in G.nodes:
                    G.add_edge((region, i), (neighbor, i))

            if i in train_indices:
                train_mask.append(True)
            else:
                train_mask.append(False)

            if i in val_indices:
                val_mask.append(True)
            else:
                val_mask.append(False)

    data = from_networkx(G)
    data.train_mask = torch.tensor(train_mask)
    data.val_mask = torch.tensor(val_mask)

    return data

def objective(trial):
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    num_hidden_channels = trial.suggest_categorical("num_hidden_channels", [4, 6, 8, 16, 32, 64, 128, 256, 512, 1024])
    num_layers = trial.suggest_categorical("num_layers", [3, 6, 9, 12, 15, 18, 21, 24, 27, 30])

    class Net(torch.nn.Module):
        def __init__(self, num_original_features, num_additional_features, num_hidden_channels, num_layers, dropout_rate):
            super(Net, self).__init__()
            self.layers = torch.nn.ModuleList()
            self.layers.append(GCNConv(num_original_features + num_additional_features, num_hidden_channels))
            for _ in range(num_layers - 2): # -2 to account for the first and last layers
                self.layers.append(GCNConv(num_hidden_channels, num_hidden_channels))
            self.layers.append(GCNConv(num_hidden_channels, num_original_features))  # output size matches num_original_features
            self.dropout_rate = dropout_rate

        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            for conv in self.layers[:-1]:
                x = conv(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = self.layers[-1](x, edge_index)  # Don't apply relu or dropout to the last layer's outputs
            return x


    results = []

    for train_index, val_index in kf.split(np.arange(window, df.shape[0] - window)):

        data = data_to_graph(df, window, train_index, val_index)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_original_features = window  # original size
        num_additional_features = 3  # new additional features
        model = Net(num_original_features, num_additional_features, num_hidden_channels, num_layers, dropout_rate).to(device)
        data = data.to(device)
        optimizer = Adam(model.parameters(), lr=lr)
        criterion = MSELoss()

        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

        model.eval()
        pred = model(data)

        mae = mean_absolute_error(data.y[data.val_mask].cpu().detach().numpy(), pred[data.val_mask].cpu().detach().numpy())
        mse = mean_squared_error(data.y[data.val_mask].cpu().detach().numpy(), pred[data.val_mask].cpu().detach().numpy())
        rmse = np.sqrt(mse)
        r2 = r2_score(data.y[data.val_mask].cpu().detach().numpy(), pred[data.val_mask].cpu().detach().numpy())
        mdape = Utils.MDAPE(data.y[data.val_mask].cpu().detach().numpy(), pred[data.val_mask].cpu().detach().numpy())

        results.append((mae, rmse, r2, mdape))

    avg_mae = np.mean([res[0] for res in results])
    avg_rmse = np.mean([res[1] for res in results])
    avg_r2 = np.mean([res[2] for res in results])
    avg_mdape = np.mean([res[3] for res in results])

    return avg_mae  # Optuna will minimize this value

# Start Optuna study
study = optuna.create_study(study_name="gcn_minimize_features_extended_15d", storage="sqlite:///gcn", load_if_exists=True, direction="minimize")
study.optimize(objective, n_trials=1000)  # Number of iterations. Increase it for better results, but it will take more time.

# Display the best parameters
print("Best parameters: ", study.best_params)
