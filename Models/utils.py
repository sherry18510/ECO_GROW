import warnings
warnings.filterwarnings('ignore')  # Ignore possible warnings

import pandas as pd  # For data processing and analysis
from tqdm import *  # For progress bars
import numpy as np  # For numerical computation and array operations
import matplotlib.pyplot as plt  # For data visualization

# Import PyTorch related libraries
import torch  # PyTorch main library
import torch.nn.functional as F  # Activation functions and other functionalities
import torch_geometric.transforms as T  # Graph data transformations

# Import machine learning libraries
from sklearn.metrics import (mean_squared_error, mean_absolute_error, 
                             mean_absolute_percentage_error, 
                             r2_score)  # Common regression metrics
from sklearn.linear_model import Lasso  # Lasso regression model
from sklearn.model_selection import GridSearchCV, KFold  # Grid search and cross validation

import os

from model import *

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
class DataSet:
    def __init__(self, base_path='..', barabasi_p=6):
        self.base_path = base_path
        self.barabasi_p = barabasi_p
        
        # Load base data
        self.data = self.load_data()
        self.target_dict = self.load_target_dict()
        self.city_id2index = self.load_city_id2index()
        self.city_index2id = self.load_city_index2id()
        self.feat_dict = self.load_feat_dict()
        
        # Load graph data
        self.cc_dist_graphs = self.load_graphs('cc_dist_graphs.pt', (2005, 2023))
        self.cc_mob_source_graphs = self.load_graphs('cc_mob_source_graphs_gaode.pt', (2019, 2022))
        self.cc_mob_destination_graphs = self.load_graphs('cc_mob_destination_graphs_gaode.pt', (2019, 2022))
        self.cc_sim_graphs = self.load_graphs('cc_sim_graphs_subtype.pt', (2005, 2023))
        self.cc_tfidf_poi_graphs = self.load_graphs('cc_tfidf_poi_graphs.pt', (2015, 2023))
        self.cc_kmeans_graphs = self.load_graphs('cc_kmeans_graphs.pt', (2019, 2023))
        self.cc_empty_graphs = self.cc_kmeans_graphs
        self.cc_barabasi_graphs = self.load_graphs('cc_barabasi_graphs.pt', (2005, 2023))
        self.cc_edge_label = self.load_cc_edge_label()
        
        print('Finish loading data √')

    def load_data(self):
        file_path = f'{self.base_path}/Data/all_2019_2021.csv'
        return pd.read_csv(file_path)

    def load_target_dict(self):
        file_path = f'{self.base_path}/Data/all_2019_2021.pkl'
        return pd.read_pickle(file_path)

    def load_city_id2index(self):
        file_path = f'{self.base_path}/Data/city_id2index.pkl'
        return pd.read_pickle(file_path)

    def load_city_index2id(self):
        file_path = f'{self.base_path}/Data/city_index2id.pkl'
        return pd.read_pickle(file_path)

    def load_feat_dict(self):
        file_path = f'{self.base_path}/GraphData/feat_dict.pkl'
        return pd.read_pickle(file_path)

    def load_graphs(self, file_name, year_range):
        year_list = list(range(*year_range))
        file_path = f'{self.base_path}/GraphData/{file_name}'
        graphs = torch.load(file_path)
        return {t: graphs[i] for i, t in enumerate(year_list)}

    def load_cc_edge_label(self):
        file_path = f'{self.base_path}/GraphData/cc_edge_label_year_threshold_p{self.barabasi_p}.pkl'
        return pd.read_pickle(file_path)
    
    
    
def evaluate_model(X, y, kf_dict=None, verbose=False):
    """
    Evaluate Lasso regression model performance.

    Parameters:
    - X: Feature matrix
    - y: Target value
    - kf_dict: KFold cross validation index dictionary
    - verbose: Whether to output detailed information

    Returns:
    - Various scores and prediction results
    """
    model = Lasso()  # Initialize Lasso regression model
    param_grid = {'alpha': [1e-6, 1e-5, 1e-4, 1e-3]}  # Set hyperparameter grid

    # Use GridSearchCV to find the best model
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_absolute_error')
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_  # Get the best model

    if kf_dict is None:
        # If KFold index is not provided, create a new KFold object
        kf = KFold(n_splits=5)
        kf_dict = {i: {'train_index': train_index, 'test_index': test_index} for i, (train_index, test_index) in enumerate(kf.split(X))}

    # Initialize score lists
    mae_scores, rmse_scores, r2_scores, mape_scores = [], [], [], []
    y_preds, y_tests = {}, {}

    for idx_list in kf_dict.values():
        train_index = idx_list['train_index']
        test_index = idx_list['test_index']

        # Split training and test sets
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train the model and make predictions
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        y_pred = [pred if pred >= 0 else 0 for pred in y_pred]

        # Calculate various scores
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2_scores.append(r2_score(y_test, y_pred))
        mape_scores.append(mean_absolute_percentage_error(y_test, y_pred))

        # Record prediction and test values
        for i, tidx in enumerate(test_index):
            y_preds[tidx] = y_pred[i]
            y_tests[tidx] = y_test[i]

    if verbose:
        # Output average scores
        print("MAE:", np.mean(mae_scores))
        print("RMSE:", np.mean(rmse_scores))
        print("R2 Score:", np.mean(r2_scores))
        print("MAPE:", np.mean(mape_scores))

    return mae_scores, rmse_scores, r2_scores, mape_scores, y_preds, y_tests, kf_dict


def evaluate_and_plot(X_dict, y_dict, fig_title, data_loader,
                      kf_dict=None, verbose=False, show_fig=True, save=False):
    """
    Evaluate model and plot results.

    Parameters:
    - X_dict: Dictionary of features for each year
    - y_dict: Dictionary of target values for each year
    - fig_title: Figure title
    - kf_dict: KFold cross validation index dictionary
    - verbose: Whether to output detailed information
    - show_fig: Whether to display the figure
    - save: Whether to save the figure

    Returns:
    - Evaluation results DataFrame
    """
    results_df = pd.DataFrame(columns=['Year', 'MAE', 'RMSE', 'R2', 'MAPE'])  # Initialize results DataFrame
    _, axs = plt.subplots(len(X_dict), 1, figsize=(20, 8))  # Create subplots

    y_pred_dict, y_test_dict = {}, {}

    for i, year in enumerate(X_dict.keys()):
        # Evaluate model
        mae_scores, rmse_scores, r2_scores, mape_scores, y_preds, y_tests, fold_indices = evaluate_model(X_dict[year], y_dict[year], kf_dict=kf_dict, verbose=verbose)

        # Record prediction and test values
        y_tests = dict(sorted(y_tests.items(), key=lambda item: item[1], reverse=False))
        y_preds = {cidx: y_preds[cidx] for cidx in y_tests.keys()}

        y_pred_dict[year] = y_preds
        y_test_dict[year] = y_tests

        # Summarize results for each year
        year_results = pd.DataFrame({'Year': [year],
                                     'MAE': [np.mean(mae_scores)],
                                     'RMSE': [np.mean(rmse_scores)],
                                     'R2': [np.mean(r2_scores)],
                                     'MAPE': [np.mean(mape_scores)]})
        results_df = pd.concat([results_df, year_results], ignore_index=True)

        # Plot test values and predicted values for each year
        axs[i].plot(list(y_tests.values()), label='test')
        axs[i].plot(list(y_preds.values()), label='pred')
        axs[i].set_title(f'Year {year}')
        axs[i].legend()

    plt.tight_layout()
    if save:
        plt.savefig(f'fig/{fig_title}.png', bbox_inches='tight')
    if show_fig:
        plt.show()
    else:
        plt.close()
    
    results_df.loc[results_df.shape[0], :] = results_df.describe().loc['mean', :]  # Add average results
    eval_df = pd.DataFrame({
        'result': results_df.RMSE.tolist() + results_df.MAE.tolist() + results_df.R2.tolist() + results_df.MAPE.tolist()
    })

    result_dict = {}
    for year, dic in y_pred_dict.items():
        temp_df = pd.DataFrame({
            'city_idx': dic.keys(),
            'pred': dic.values(),
            'label': y_test_dict[year].values()
        })
        temp_df['year'] = year
        temp_df['city_name'] = temp_df.city_idx.map(data_loader.city_index2id)

        result_dict[year] = temp_df
    
    return results_df, result_dict


def extract_weighted_adjacency_matrix(edge_index, edge_weight, num_nodes):
    # Create sparse adjacency matrix
    adj_matrix = torch.sparse.FloatTensor(edge_index, edge_weight, torch.Size([num_nodes, num_nodes]))
    adj_matrix = adj_matrix.to_dense()  # Convert to dense tensor
    return adj_matrix


def train_pred(file_name, layer_name='TKGCN',
               loss_name='BCE', alpha=0.6, beta=0.5, weight_decay=1e-4, barabasi_p=6,
               num_epochs=250, num_layer=3, lr=0.01, dropout_rate=0.15, 
               return_embed=False, top_k=5, device='cuda'):
    graph_list = [
        'cc_dist_graphs', 'cc_sim_graphs', 
        'cc_tfidf_poi_graphs', 'cc_mob_source_graphs', 
        'cc_mob_destination_graphs', 'cc_empty_graphs'
    ]
    graph_cnt = len(graph_list)
    with_empty = 'cc_empty_graphs' in graph_list
    
    # ------------------------------------ Load Data ------------------------------------ #
    data_loader = DataSet(barabasi_p=barabasi_p)

    save_dir = f'result/{file_name}/{layer_name}_{loss_name}_{num_epochs}_{alpha}_{dropout_rate}'
    os.makedirs(save_dir, exist_ok=True)
    
    # ------------------------------------ Train Embedding ------------------------------------ #
    embedding_dict = {}  # Store node embeddings in dictionary

    # Initialize loss dictionary
    loss_dict = {t: [] for t in range(2019, 2022)}
    loss1_dict = {t: [] for t in range(2019, 2022)}
    loss2_dict = {t: [] for t in range(2019, 2022)}
    loss3_dict = {t: [] for t in range(2019, 2022)}

    criterion1 = torch.nn.BCELoss()  # Mean squared error loss
    criterion2 = torch.nn.BCEWithLogitsLoss()
    criterion3 = torch.nn.BCELoss()  
    
    print('Begin training >>>>>>')
    # Train model
    for timestamp in range(2019, 2022):
        # Initialize model
        model = ECO_GROW(
                        in_channels=16, 
                        node_feat_dim=3, 
                        layer_name=layer_name,
                        graph_cnt=graph_cnt, 
                        num_layers=num_layer, 
                        out_channels=16, 
                        with_empty=with_empty, 
                        hidden_channels=128,
                        top_k=top_k,
                        dropout_rate=dropout_rate,
                        device=device).to(device)

        # Initialize optimizer and loss function
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)


        # Get node features and graph structure
        node_feature = data_loader.feat_dict[timestamp]
        node_feature = torch.tensor(node_feature, dtype=torch.float32).to(device)
        static_graphs = [
            data_loader.cc_dist_graphs[timestamp].to(device),
            data_loader.cc_sim_graphs[timestamp].to(device),
            data_loader.cc_tfidf_poi_graphs[timestamp].to(device),
            data_loader.cc_mob_source_graphs[timestamp].to(device),
            data_loader.cc_mob_destination_graphs[timestamp].to(device),
            data_loader.cc_empty_graphs[timestamp].to(device),
        ]
        dynamic_graphs = [data_loader.cc_sim_graphs[year].to(device) for year in range(timestamp - 14, timestamp + 1)]

        # Prepare potential edge indices and node features
        target1 = [1 if g >=0 else 0 for g in data_loader.target_dict[timestamp]['yoy_gr']]
        target1 = torch.tensor(target1).to(device).unsqueeze(1)  # Convert to tensor and move to device

        # Prepare potential edge indices and node features
        potential_edge_index = data_loader.cc_barabasi_graphs[timestamp].edge_index
        target2 = torch.tensor(data_loader.cc_edge_label[timestamp]).to(device)
        
        target3 = [1 if g >= 0 else 0 for g in data_loader.target_dict[timestamp]['job_gr']]
        target3 = torch.tensor(target3).to(device).unsqueeze(1)

        # Train model
        for epoch in trange(num_epochs):
            model.train()  # Set model to training mode
            optimizer.zero_grad()  # Zero gradients

            # Forward propagation
            output, embedding = model(node_feature, static_graphs, dynamic_graphs, potential_edge_index)
            output1, output2, output3 = output[0], output[1], output[2]


            loss1 = criterion1(output1.float(), target1.float())
            loss2 = criterion2(output2.float(), target2.float())
            loss3 = criterion3(output3.float(), target3.float())

            loss = beta * (alpha * loss1 + (1-alpha) * loss3) + (1-beta) * loss2

            loss_dict[timestamp].append(float(loss))
            loss1_dict[timestamp].append(float(loss1))
            loss2_dict[timestamp].append(float(loss2))
            loss3_dict[timestamp].append(float(loss3))
            

            loss.backward()  # Backward propagation
            optimizer.step()  # Update model parameters

            if epoch == num_epochs - 1:  # Save node embeddings in the last epoch
                embedding_dict[timestamp] = embedding

    # Clear GPU cache
    torch.cuda.empty_cache()

    # Plot loss curve
    plt.figure(figsize=(20, 4))
    for timestamp, loss_list in loss_dict.items():
        plt.plot(loss_list, marker='o', label=timestamp)  # Plot loss curve for each timestamp
    plt.legend()
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    # ------------------------------------ Test Embedding Downstream 1------------------------------------ #
    print('Evaluating >>>>>>')
    timestamp_list = list(range(2019, 2022))  # Timestamp list

    X_dict = {}  # Store features in dictionary
    y_dict = {}  # Store labels in dictionary

    for t_idx, timestamp in tqdm(enumerate(timestamp_list[:])):
        X_dict[timestamp] = []  # Initialize feature list
        y_dict[timestamp] = []  # Initialize label list

        # Add y label
        time_cnt = data_loader.data.query(f'year == {timestamp_list[t_idx]}')  # Query data for specific year
        cnt_dict = {data_loader.city_id2index[c]: n for c, n in zip(time_cnt.city_name.tolist(), time_cnt.city_new_comp_cnt.tolist())}  # CityID and new company count dictionary

        cnt_list = []  # New company count list
        embeddings = np.concatenate((
            embedding_dict[timestamp].cpu().detach().numpy(),
            #data_loader.feat_dict[timestamp]
            np.array(data_loader.target_dict[timestamp]['GDP_norm']).reshape(-1, 1),
            np.array(data_loader.target_dict[timestamp]['population_norm']).reshape(-1, 1),
            np.array(data_loader.target_dict[timestamp]['city_new_comp_cnt_last']).reshape(-1, 1),
            np.array(data_loader.target_dict[timestamp]['job_count_last']).reshape(-1, 1),
        ), axis=1)  # Merge node embeddings and other features

        for i in range(297):
            cnt = cnt_dict.get(i, 0)
            if cnt >= 0:
                cnt_list.append(cnt)  # Add new company count

        X_dict[timestamp] = np.array(embeddings)  # Store features
        y_dict[timestamp] = np.array(cnt_list)  # Store labels

    # Use function for evaluation and plotting
    results, _ = evaluate_and_plot(X_dict, y_dict, '', data_loader=data_loader, show_fig=True)

    # ---------------------------- Print Graph Scores ---------------------------- #
    model_state_dict = model.state_dict()  # Get model state dictionary

    # Get score parameters for each graph
    graph_scores = model_state_dict['graph_scores']  # Get graph score parameters

    # Convert parameters to numpy array for viewing
    graph_scores_numpy = F.softmax(graph_scores.cpu().detach(), dim=0)  # Use softmax to calculate importance
    
    # Create dictionary to save each graph score
    scores_dict = {graph_list[i]: graph_scores_numpy[i] for i in range(len(graph_list))}

    # Output each graph score
    for i, score in enumerate(graph_scores_numpy):
        print(f'Graph {graph_list[i]} score: {score}')  # Print each graph score
    
    torch.save(model_state_dict, f'{save_dir}/model.pth')
    pd.to_pickle(embedding_dict, f'{save_dir}/embed.pkl')  # Save results
    pd.to_pickle(X_dict, f'{save_dir}/X.pkl')  # Save results
    pd.to_pickle(scores_dict, f'{save_dir}/scores_dict.pkl')  # Save results
    results.to_csv(f'{save_dir}/results.csv', index=False)
    
    print(results)

# ------------------------------------ Test Embedding Downstream 2 ------------------------------------ #
    print('Evaluating >>>>>>')
    timestamp_list = list(range(2019, 2022))  # Timestamp list

    X_dict = {}  # Store features in dictionary
    y_dict = {}  # Store labels in dictionary

    for t_idx, timestamp in tqdm(enumerate(timestamp_list[:])):
        X_dict[timestamp] = []  # Initialize feature list
        y_dict[timestamp] = []  # Initialize label list

        # Add y label
        time_cnt = data_loader.data.query(f'year == {timestamp_list[t_idx]}')  # Query data for specific year
        cnt_dict = {data_loader.city_id2index[c]: n for c, n in zip(time_cnt.city_name.tolist(), time_cnt.job_count.tolist())}  # CityID and new company count dictionary

        cnt_list = []  # New company count list
        embeddings = np.concatenate((
            embedding_dict[timestamp].cpu().detach().numpy(),
            #data_loader.feat_dict[timestamp]
            np.array(data_loader.target_dict[timestamp]['GDP_norm']).reshape(-1, 1),
            np.array(data_loader.target_dict[timestamp]['population_norm']).reshape(-1, 1),
            np.array(data_loader.target_dict[timestamp]['city_new_comp_cnt_last']).reshape(-1, 1),
            np.array(data_loader.target_dict[timestamp]['job_count_last']).reshape(-1, 1),
        ), axis=1)  # Merge node embeddings and other features

        for i in range(297):
            cnt = cnt_dict.get(i, 0)
            if cnt >= 0:
                cnt_list.append(cnt)  # Add new company count

        X_dict[timestamp] = np.array(embeddings)  # Store features
        y_dict[timestamp] = np.array(cnt_list)  # Store labels

    # Use function for evaluation and plotting
    results, _ = evaluate_and_plot(X_dict, y_dict, '', data_loader=data_loader, show_fig=True)

    # ---------------------------- Print Graph Scores ---------------------------- #
    model_state_dict = model.state_dict()  # Get model state dictionary

    # Get score parameters for each graph
    graph_scores = model_state_dict['graph_scores']  # Get graph score parameters

    # Convert parameters to numpy array for viewing
    graph_scores_numpy = F.softmax(graph_scores.cpu().detach(), dim=0)  # Use softmax to calculate importance
    
    # Create dictionary to save each graph score
    scores_dict = {graph_list[i]: graph_scores_numpy[i] for i in range(len(graph_list))}

    # Output each graph score
    for i, score in enumerate(graph_scores_numpy):
        print(f'Graph {graph_list[i]} score: {score}')  # Print each graph score
    
    torch.save(model_state_dict, f'{save_dir}/model.pth')
    pd.to_pickle(embedding_dict, f'{save_dir}/embed.pkl')  # Save results
    pd.to_pickle(X_dict, f'{save_dir}/X.pkl')  # Save results
    pd.to_pickle(scores_dict, f'{save_dir}/scores_dict.pkl')  # Save results
    results.to_csv(f'{save_dir}/results.csv', index=False)
    
    print(results)
