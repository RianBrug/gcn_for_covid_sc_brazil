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
from torch.nn import MSELoss
import torch.nn.functional as F
from torch.optim import Adam
from utils import Utils
import numpy as np
from sklearn.model_selection import KFold
import optuna
import sqlite3
import optuna.visualization as vis
from torch_geometric.nn import GATConv
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    # num_hidden_channels = trial.suggest_categorical("num_hidden_channels", [16, 32, 64, 128, 256, 512, 1024])
    num_hidden_channels = trial.suggest_categorical("num_hidden_channels", [1, 2, 4, 6, 8, 10, 12, 14, 16])
    # num_layers = trial.suggest_categorical("num_layers", [6, 9, 12, 15, 18, 21, 24])
    num_layers = trial.suggest_categorical("num_layers", [1, 2, 3, 4, 5, 6, 7, 8, 9])
    weight_decay = trial.suggest_float("weight_decay", 1e-10, 1e-3)  # L2 regularization
    num_heads = trial.suggest_categorical("num_heads", [1, 2, 4, 8, 16])  # Number of attention heads

    class Net(torch.nn.Module):
        def __init__(self, num_original_features, num_additional_features, num_hidden_channels, num_layers, dropout_rate, num_heads):
            super(Net, self).__init__()
            self.layers = torch.nn.ModuleList()
            self.layers.append(GATConv(num_original_features + num_additional_features, num_hidden_channels, heads=num_heads))
            for _ in range(num_layers - 2): # -2 to account for the first and last layers
                self.layers.append(GATConv(num_hidden_channels * num_heads, num_hidden_channels, heads=num_heads))
            self.layers.append(GATConv(num_hidden_channels * num_heads, num_original_features))  # output size matches num_original_features
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
    total_epochs = 100
    

    for fold, (train_index, val_index) in enumerate(kf.split(np.arange(window, df.shape[0] - window))):
        data = data_to_graph(df, window, train_index, val_index)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_original_features = window  # original size
        num_additional_features = 3  # new additional features
        model = Net(num_original_features, num_additional_features, num_hidden_channels, num_layers, dropout_rate, num_heads).to(device)
        data = data.to(device)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # Added weight decay for L2 regularization
        scheduler = ReduceLROnPlateau(optimizer, 'min')  # Added a learning rate scheduler
        criterion = MSELoss()

        model.train()
        fold_losses = []
        for epoch in range(total_epochs):
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            fold_losses.append(loss.item())
            scheduler.step(loss)  # Decrease lr if the loss plateaus

        avg_fold_loss = sum(fold_losses) / len(fold_losses)

        # Pass the average fold loss to the pruner
        unique_epoch = fold * total_epochs + epoch
        trial.report(avg_fold_loss, unique_epoch)

        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        model.eval()
        pred = model(data)

        mae = mean_absolute_error(data.y[data.val_mask].cpu().detach().numpy(), pred[data.val_mask].cpu().detach().numpy())
        mse = mean_squared_error(data.y[data.val_mask].cpu().detach().numpy(), pred[data.val_mask].cpu().detach().numpy())
        rmse = np.sqrt(mse)
        r2 = r2_score(data.y[data.val_mask].cpu().detach().numpy(), pred[data.val_mask].cpu().detach().numpy())
        mdape = Utils.MDAPE(data.y[data.val_mask].cpu().detach().numpy(), pred[data.val_mask].cpu().detach().numpy())

        results.append((mae, mse, rmse, r2, mdape))

    avg_mae = np.mean([res[0] for res in results])
    avg_mse = np.mean([res[1] for res in results])
    avg_rmse = np.mean([res[2] for res in results])
    avg_r2 = np.mean([res[3] for res in results])
    avg_mdape = np.mean([res[4] for res in results])

    trial.set_user_attr("avg_rmse", float(avg_rmse))
    trial.set_user_attr("avg_r2", float(avg_r2))
    trial.set_user_attr("avg_mdape", float(avg_mdape))
    trial.set_user_attr("avg_mse", float(avg_mse))
    return avg_mae  # Optuna will minimize this value

# Start Optuna study
n_trails = 1000
pruner = optuna.pruners.MedianPruner()
study = optuna.create_study(study_name="GAT_network_smaller", storage="sqlite:///gcn", load_if_exists=True, direction="minimize", pruner=pruner)
study.optimize(objective, n_trials=n_trails)  # Number of iterations. Increase it for better results, but it will take more time.
vis.plot_optimization_history(study)
vis.plot_intermediate_values(study)
vis.plot_parallel_coordinate(study)
vis.plot_slice(study)
vis.plot_param_importances(study)

# Plot optimization history
plot = vis.plot_optimization_history(study)
plt.title("Optimization History")
plt.savefig("optimization_history.png")

# Plot intermediate values
plt.figure()  # create a new figure
plot = vis.plot_intermediate_values(study)
plt.title("Intermediate Values")
plt.savefig("intermediate_values.png")

# Plot high-dimensional parameter relationships
plt.figure()  # create a new figure
plot = vis.plot_parallel_coordinate(study)
plt.title("High-dimensional Parameter Relationships")
plt.savefig("parameter_relationships.png")

# Plot parameters
plt.figure()  # create a new figure
plot = vis.plot_slice(study)
plt.title("Parameters")
plt.savefig("parameters.png")

# Plot parameter importances
plt.figure()  # create a new figure
plot = vis.plot_param_importances(study)
plt.title("Parameter Importances")
plt.savefig("param_importances.png")
# Display the best parameters
print("Best parameters: ", study.best_params)

for trial in study.trials:
    print("Trial", trial.number)
    print("MAE", trial.value)
    print("MSE", trial.user_attrs["avg_mse"])
    print("RMSE", trial.user_attrs["avg_rmse"])
    print("R2", trial.user_attrs["avg_r2"])
    print("MDAPE", trial.user_attrs["avg_mdape"])