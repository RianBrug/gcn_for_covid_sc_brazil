import json
import sklearn
import torch
import networkx as nx
import pandas as pd
import torch_geometric
from torch_geometric.utils import from_networkx
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from torch_geometric.nn import GCNConv
from torch.nn import L1Loss
import torch.nn.functional as F
from torch.optim import Adam
import torch.multiprocessing as mp
import numpy as np
import optuna
import sqlite3
import optuna.visualization as vis
from optuna.pruners import SuccessiveHalvingPruner
from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib.pyplot as plt
from earlyStopping import EarlyStopping
from utils import Utils

# Configuration
HOME = '/Users/rcvb/Documents/tcc_rian/code'
DATABASE_URL = 'sqlite:///gcn_newest'
STUDY_NAME = 'gcn_newest'
WINDOW = 15
TOTAL_EPOCHS = 160
TRIALS_UNTIL_START_PRUNING = 150
N_TRIALS = 5
N_JOBS = 5
NUM_ORIGINAL_FEATURES = 15
NUM_ADDITIONAL_FEATURES = 1
PATIENCE_LEARNING_SCHEDULER = 130
NUM_SPLITS = 3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
early_stopping = EarlyStopping(patience=75, verbose=True)

# Define the Neural Network Model
class Net(torch.nn.Module):
    def __init__(self, NUM_ORIGINAL_FEATURES, NUM_ADDITIONAL_FEATURES, num_hidden_channels, num_layers, dropout_rate):
        super(Net, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(NUM_ORIGINAL_FEATURES + NUM_ADDITIONAL_FEATURES, num_hidden_channels))
        for _ in range(num_layers - 2): # -2 to account for the first and last layers
            self.layers.append(GCNConv(num_hidden_channels, num_hidden_channels))
        self.layers.append(GCNConv(num_hidden_channels, NUM_ORIGINAL_FEATURES))  # output size matches num_original_features
        self.dropout_rate = dropout_rate

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.layers[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            if self.training:  # only apply dropout during training
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.layers[-1](x, edge_index)  # Don't apply relu or dropout to the last layer's outputs
        return x

    
# load and return the dataframe
def load_data():
    HOME = '/Users/rcvb/Documents/tcc_rian/code'
    with open(f'{HOME}/assets/confirmed_cases_by_region_and_date.json') as file:
        data = json.load(file)

    df = pd.DataFrame(data)
    df.reset_index(inplace=True)
    df.rename(columns={'index':'collect_date'}, inplace=True)
    df['collect_date'] = pd.to_datetime(df['collect_date'])
    df.sort_values(by=['collect_date'], inplace=True)
    df.drop(df.columns[len(df.columns)-1], axis=1, inplace=True)
    df.fillna(0, inplace=True)
    return df
    
def sliding_windows(data, window):
    X = []
    Y = []

    for i in range(len(data)-2*window):
        X.append(data.iloc[i:i+window].values)
        Y.append(data.iloc[i+window:i+2*window].values) 

    return np.array(X), np.array(Y)

# Function to convert data into graphs
def data_to_graph(df, window, train_indices, val_indices):
    G = nx.Graph()
    train_mask = []
    val_mask = []

    # Load your additional features data here. For example:
    pr_df = pd.read_csv(f'{HOME}/assets/populacao_residente_sc_por_macroregiao.csv', sep=";", index_col=0)


    for region in df.columns[1:]:
        region_df = df[['collect_date', region]].dropna()
        X, Y = sliding_windows(region_df[region], window)
        
        # Retrieve additional features for the current region
        add_features = np.array([
            pr_df.loc[region],
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

# Objective function for Optuna
def objective(trial):
    df = load_data()
    tscv = TimeSeriesSplit(n_splits=3)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-1,log=True)
    num_hidden_channels = trial.suggest_categorical("num_hidden_channels", [16, 32, 64, 96, 128, 256])
    num_layers = trial.suggest_categorical("num_layers", [6, 9, 12, 15, 18, 21])
    #num_hidden_channels = trial.suggest_categorical("num_hidden_channels", [5, 15, 30, 75, 100, 150, 250])
    #num_layers = trial.suggest_categorical("num_layers", [5, 10, 15, 25, 40, 50])
    weight_decay = trial.suggest_float("weight_decay", 1e-10, 1e-3)  # L2 regularization
    
    results = []
    true_values = []
    predictions = []

    for fold, (train_index, val_index) in enumerate(tscv.split(np.arange(WINDOW, df.shape[0] - WINDOW))):
        data = data_to_graph(df, WINDOW, train_index, val_index)
        model = Net(NUM_ORIGINAL_FEATURES, NUM_ADDITIONAL_FEATURES, num_hidden_channels, num_layers, dropout_rate).to(DEVICE)
        data = data.to(DEVICE)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # Added weight decay for L2 regularization
        # optimizer = RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=PATIENCE_LEARNING_SCHEDULER)  # Added a learning rate scheduler
        criterion = L1Loss()
        model.train()
        fold_losses = []  # Average loss for each epoch within this fold	
        val_losses = []  # Validation loss for each epoch within this fold
        for epoch in range(TOTAL_EPOCHS):
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            model.eval()
            fold_losses.append(loss.item())

            with torch.no_grad():
                pred = model(data)
                val_loss = criterion(pred[data.val_mask], data.y[data.val_mask])
                val_losses.append(val_loss.item())
                true_values = data.y[data.val_mask].cpu().detach().numpy().tolist()
                predictions = pred[data.val_mask].cpu().detach().numpy().tolist()

            scheduler.step(val_loss)

            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        model.load_state_dict(torch.load('checkpoint.pt'))
        avg_fold_loss = sum(fold_losses) / len(fold_losses)

        if trial.number > TRIALS_UNTIL_START_PRUNING:
            # Pass the average fold loss to the pruner
            unique_epoch = fold * TOTAL_EPOCHS + epoch
            trial.report(avg_fold_loss, unique_epoch)

            # Handle pruning based on the intermediate value
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        if np.isnan(data.y[data.val_mask].cpu().detach().numpy()).any():
            print("NaN value detected in target data.")
            return np.inf  # Optuna will minimize this value
        
        # Check for NaN values in prediction
        if np.isnan(pred[data.val_mask].cpu().detach().numpy()).any():
            print("NaN value detected in prediction.")
            return np.inf  # Optuna will minimize this value

        mae = mean_absolute_error(data.y[data.val_mask].cpu().detach().numpy(), pred[data.val_mask].cpu().detach().numpy())
        mape = mean_absolute_percentage_error(data.y[data.val_mask].cpu().detach().numpy(), pred[data.val_mask].cpu().detach().numpy())
        mse = mean_squared_error(data.y[data.val_mask].cpu().detach().numpy(), pred[data.val_mask].cpu().detach().numpy())
        rmse = np.sqrt(mse)
        r2 = r2_score(data.y[data.val_mask].cpu().detach().numpy(), pred[data.val_mask].cpu().detach().numpy())
        mdape = Utils.MDAPE(data.y[data.val_mask].cpu().detach().numpy(), pred[data.val_mask].cpu().detach().numpy())

        results.append((mae, mape, mse, rmse, r2, mdape, val_losses[-33:]))

    avg_mae = np.mean([res[0] for res in results])
    avg_mape = np.mean([res[1] for res in results])
    avg_mse = np.mean([res[2] for res in results])
    avg_rmse = np.mean([res[3] for res in results])
    avg_r2 = np.mean([res[4] for res in results])
    avg_mdape = np.mean([res[5] for res in results])
    avg_val_losses = np.mean([res[6] for res in results])

    trial.set_user_attr("avg_mae", float(avg_mae))
    trial.set_user_attr("avg_mape", float(avg_mape))
    trial.set_user_attr("avg_mse", float(avg_mse))
    trial.set_user_attr("avg_rmse", float(avg_rmse))
    trial.set_user_attr("avg_r2", float(avg_r2))
    trial.set_user_attr("avg_mdape", float(avg_mdape))
    trial.set_user_attr("avg_val_losses", float(avg_val_losses))
    trial.set_user_attr("true_values", true_values)
    trial.set_user_attr("predictions", predictions)

    return avg_mae  # Optuna will minimize this value

def plot_results(true_values, predictions):
    plt.figure(figsize=(12, 8))
    plt.plot(true_values, label='True values')
    plt.plot(predictions, label='Predictions')
    plt.xlabel('Time Steps')
    plt.ylabel('Confirmed Cases')
    plt.legend()
    plt.show()

# Main part of the code
if __name__ == '__main__':
    # Enable parallel processing
    mp.set_start_method('spawn')
    
    df = load_data()
    
    # Split data into training and testing sets to prevent data leakage
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)

    
    # Optuna study for hyperparameter tuning
    # Start Optuna study
    pruner = SuccessiveHalvingPruner()
    study = optuna.create_study(study_name=STUDY_NAME, storage=DATABASE_URL, load_if_exists=True, direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS, show_progress_bar=True)
    vis.plot_optimization_history(study)
    vis.plot_intermediate_values(study)
    vis.plot_parallel_coordinate(study)
    vis.plot_slice(study)
    vis.plot_param_importances(study)
    vis.plot_edf(study)
    vis.plot_contour(study)    
    
    # Plot results
    best_trial = study.best_trial
    best_true_values = np.array(best_trial.user_attrs["true_values"])
    best_predictions = np.array(best_trial.user_attrs["predictions"])
    plot_results(best_true_values, best_predictions)
