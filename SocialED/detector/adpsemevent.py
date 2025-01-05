import networkx as nx
from itertools import combinations, chain
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
from sklearn.cluster import SpectralClustering
import sys
from datetime import datetime
import math
import pickle
import pandas as pd
import os
from os.path import exists
import time
import multiprocessing
import torch
from matplotlib import pyplot as plt
from networkx.algorithms import cuts
from sentence_transformers import SentenceTransformer
import re
from sklearn.model_selection import train_test_split
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ADPSEMEvent():
    def __init__(self, dataset):
        self.dataset = dataset
        self.language = dataset.get_dataset_language()
        self.dataset_name = dataset.get_dataset_name()
        self.save_path = "../model/model_saved/adpsemevent/"+self.dataset_name+"/"

    def preprocess(self):
        preprocessor = Preprocessor(self.dataset)
        preprocessor.preprocess()

    def detection(self):
        ground_truths, predictions = run_hier_2D_SE_mini_closed_set(self.save_path, n=300, e_a=True, e_s=True)
        return ground_truths, predictions

    def evaluate(self, ground_truths, predictions):
        """
        Evaluate the model.
        """
        # Calculate Normalized Mutual Information (NMI)
        nmi = metrics.normalized_mutual_info_score(ground_truths, predictions)
        print(f"Normalized Mutual Information (NMI): {nmi}")

        # Calculate Adjusted Mutual Information (AMI)
        ami = metrics.adjusted_mutual_info_score(ground_truths, predictions)
        print(f"Adjusted Mutual Information (AMI): {ami}")

        # Calculate Adjusted Rand Index (ARI)
        ari = metrics.adjusted_rand_score(ground_truths, predictions)
        print(f"Adjusted Rand Index (ARI): {ari}")

class Preprocessor:
    def __init__(self, dataset, mode='close'):
        """Initialize preprocessor
        Args:
            dataset: Dataset calss (e.g. Event2012, Event2018, etc.)
            language: Language of the dataset (default 'English')
            mode: 'open' or 'close' (default 'close') - determines preprocessing mode
        """
        self.dataset = dataset
        self.language = dataset.get_dataset_language()
        self.dataset_name = dataset.get_dataset_name()
        self.mode = mode
        self.columns = ['tweet_id', 'text', 'event_id', 'words', 'filtered_words',
                       'entities', 'user_id', 'created_at', 'urls', 'hashtags', 'user_mentions']

    def get_closed_set_test_df(self, df):
        """Get closed set test dataframe"""
        save_path = f'../model/model_saved/adpsemevent/{self.dataset_name}/closed_set/'
        if not exists(save_path):
            os.makedirs(save_path)
        
        test_set_df_np_path = save_path + 'test_set.npy'
        if not exists(test_set_df_np_path):
            # Use 2012-style processing for all datasets
            test_mask = torch.load(f'../model/model_saved/adpsemevent/{self.dataset_name}/masks/test_mask.pt').cpu().detach().numpy()
            test_mask = list(np.where(test_mask==True)[0])
            test_df = df.iloc[test_mask]
            
            test_df_np = test_df.to_numpy()
            np.save(test_set_df_np_path, test_df_np)
        return

    def get_closed_set_messages_embeddings(self):
        """Get SBERT embeddings for closed set messages"""
        save_path = f'../model/model_saved/adpsemevent/{self.dataset_name}/closed_set/'
        
        SBERT_embedding_path = f'{save_path}/SBERT_embeddings.pkl'
        if not exists(SBERT_embedding_path):
            test_set_df_np_path = save_path + 'test_set.npy'
            test_df_np = np.load(test_set_df_np_path, allow_pickle=True)
            
            test_df = pd.DataFrame(data=test_df_np, columns=self.columns)
            print("Dataframe loaded.")

            processed_text = [preprocess_sentence(s) for s in test_df['text'].values]
            print('message text contents preprocessed.')

            embeddings = SBERT_embed(processed_text, language=self.language)

            with open(SBERT_embedding_path, 'wb') as fp:
                pickle.dump(embeddings, fp)
            print('SBERT embeddings stored.')
        return

    def get_open_set_messages_embeddings(self):
        """Get SBERT embeddings for open set messages"""
        save_path = f'../model/model_saved/adpsemevent/{self.dataset_name}/open_set/'
        num_blocks = 21  # Use 2012-style processing for all datasets
        
        for i in range(num_blocks):
            block = i + 1
            print('\n\n====================================================')
            print('block: ', block)

            SBERT_embedding_path = f'{save_path}{block}/SBERT_embeddings.pkl'

            if not exists(SBERT_embedding_path):
                df_np = np.load(f'{save_path}{block}/{block}.npy', allow_pickle=True)
                
                df = pd.DataFrame(data=df_np, columns=self.columns + ['original_index', 'date'])
                print("Dataframe loaded.")

                df['processed_text'] = [preprocess_sentence(s) for s in df['text']]
                print('message text contents preprocessed.')

                embeddings = SBERT_embed(df['processed_text'].tolist(), language=self.language)

                with open(SBERT_embedding_path, 'wb') as fp:
                    pickle.dump(embeddings, fp)
                print('SBERT embeddings stored.')
        return

    def split_open_set(self, df, root_path):
        """Split data into open set blocks"""
        if not exists(root_path):
            os.makedirs(root_path)
        
        df = df.sort_values(by='created_at').reset_index()
        df['date'] = [d.date() for d in df['created_at']]

        distinct_dates = df.date.unique()

        # First week -> block 0
        folder = root_path + '0/'
        if not exists(folder):
            os.mkdir(folder)
            
        df_np_path = folder + '0.npy'
        if not exists(df_np_path):
            ini_df = df.loc[df['date'].isin(distinct_dates[:7])]
            ini_df_np = ini_df.to_numpy()
            np.save(df_np_path, ini_df_np)

        # Following dates -> block 1, 2, ...
        end = len(distinct_dates) - 1  # Use 2012-style processing
        for i in range(7, end):
            folder = root_path + str(i - 6) + '/'
            if not exists(folder):
                os.mkdir(folder)
            
            df_np_path = folder + str(i - 6) + '.npy'
            if not exists(df_np_path):
                incr_df = df.loc[df['date'] == distinct_dates[i]]
                incr_df_np = incr_df.to_numpy()
                np.save(df_np_path, incr_df_np)
        return

    def preprocess(self):
        """Main preprocessing function"""
        # Load raw data using 2012-style processing
        df_np = self.dataset.load_data()
        
        print("Loaded data.")
        df = pd.DataFrame(data=df_np, columns=self.columns)
        print("Data converted to dataframe.")

        
        if self.mode == 'open':
            # Open-set setting
            root_path = f'../model/model_saved/adpsemevent/{self.dataset_name}/open_set/'
            self.split_open_set(df, root_path)
            self.get_open_set_messages_embeddings()
        else:
            # Close-set setting
            # Create masks directory and generate train/val/test splits
            save_dir = os.path.join(f'../model/model_saved/adpsemevent/{self.dataset_name}', 'masks')
            os.makedirs(save_dir, exist_ok=True)
            
            # Split and save masks
            self.split_and_save_masks(df, save_dir)
            print("Generated and saved train/val/test masks.")

            self.get_closed_set_test_df(df)
            self.get_closed_set_messages_embeddings()
        
        return

    def split_and_save_masks(self, df, save_dir, train_size=0.7, val_size=0.1, test_size=0.2, random_seed=42):
        """
        Splits the DataFrame into training, validation, and test sets, and saves the indices (masks) as .pt files.
        
        Parameters:
        - df (pd.DataFrame): The DataFrame to be split
        - save_dir (str): Directory to save the masks
        - train_size (float): Proportion for training (default 0.7)
        - val_size (float): Proportion for validation (default 0.1) 
        - test_size (float): Proportion for testing (default 0.2)
        - random_seed (int): Random seed for reproducibility
        """
        if train_size + val_size + test_size != 1.0:
            raise ValueError("train_size + val_size + test_size must equal 1.0")

        if df.empty:
            raise ValueError("The input DataFrame is empty.")

        print(f"Total samples in DataFrame: {len(df)}")
        
        # Set random seed
        torch.manual_seed(random_seed)

        # Split into train and temp
        train_data, temp_data = train_test_split(df, train_size=train_size, random_state=random_seed)
        
        # Split temp into val and test
        val_data, test_data = train_test_split(temp_data, 
                                             train_size=val_size/(val_size + test_size),
                                             random_state=random_seed)

        # Create boolean masks
        full_train_mask = torch.zeros(len(df), dtype=torch.bool)
        full_val_mask = torch.zeros(len(df), dtype=torch.bool)
        full_test_mask = torch.zeros(len(df), dtype=torch.bool)

        # Set indices
        full_train_mask[train_data.index] = True
        full_val_mask[val_data.index] = True  
        full_test_mask[test_data.index] = True

        print(f"Training samples: {full_train_mask.sum()}")
        print(f"Validation samples: {full_val_mask.sum()}")
        print(f"Test samples: {full_test_mask.sum()}")

        # Save masks
        mask_paths = {
            'train_mask.pt': full_train_mask,
            'val_mask.pt': full_val_mask, 
            'test_mask.pt': full_test_mask
        }

        for filename, mask in mask_paths.items():
            mask_path = os.path.join(save_dir, filename)
            if not os.path.exists(mask_path):
                try:
                    torch.save(mask, mask_path)
                    print(f"Saved {filename}")
                except Exception as e:
                    print(f"Error saving {filename}: {e}")
            else:
                print(f"{filename} already exists")

            # Verify saved file
            if os.path.exists(mask_path):
                saved_mask = torch.load(mask_path)
                if saved_mask.numel() == 0:
                    print(f"Warning: {filename} is empty")
                else:
                    print(f"Verified {filename} with {saved_mask.numel()} elements")

        print("Mask generation completed")

def get_stable_point(path, if_updata, epsilon):
    stable_point_path = path + f'stable_point_{epsilon}.pkl'
    if not exists(stable_point_path) or if_updata == True:
        embeddings_path = path + 'SBERT_embeddings.pkl'
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        first_stable_point, global_stable_point, Sensitivity = search_stable_points(embeddings, epsilon, path)
        stable_points = {'first': first_stable_point, 'global': global_stable_point}
        with open(stable_point_path, 'wb') as fp:
            pickle.dump(stable_points, fp)
        print('stable points stored.')

    with open(stable_point_path, 'rb') as f:
        stable_points = pickle.load(f)
    print('stable points loaded.')
    return stable_points, Sensitivity

def run_hier_2D_SE_mini_open_set(save_path, n=400, e_a=True, e_s=True, test_with_one_block=True, epsilon=0.2):
    if test_with_one_block:
        blocks = [16]
    else:
        blocks = [i+1 for i in range(20) if i+1>=1]
        
    for block in blocks:
        print('\n\n====================================================')
        print('block: ', block)
        print(datetime.now().strftime("%H:%M:%S"))

        folder = f'{save_path}{block}/'
        
        # Load message embeddings
        embeddings_path = folder + 'SBERT_embeddings.pkl'
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        
        # Load and process dataframe
        df_np = np.load(f'{folder}{block}.npy', allow_pickle=True)
        columns = ['tweet_id', 'text', 'event_id', 'words', 'filtered_words',
                  'entities', 'user_id', 'created_at', 'urls', 'hashtags', 'user_mentions']
        df = pd.DataFrame(data=df_np, columns=columns)
        
        all_node_features = [list(set([str(u)] + \
            [str(each) for each in um] + \
            [h.lower() for h in hs] + \
            e)) \
            for u, um, hs, e in \
            zip(df['user_id'], df['user_mentions'], df['hashtags'], df['entities'])]
        
        start_time = time.time()
        stable_points, Sensitivity = get_stable_point(folder, if_updata=True, epsilon=epsilon)
        if e_a == False: # only rely on e_s (semantic-similarity-based edges)
            default_num_neighbors = stable_points['global']
        else:
            default_num_neighbors = stable_points['first']
        if default_num_neighbors == 0:
            default_num_neighbors = math.ceil((len(embeddings)/1000)*10)
            
        global_edges = get_global_edges(all_node_features, epsilon, folder, default_num_neighbors, e_a=e_a, e_s=e_s)
        
        corr_matrix = np.load(f"{folder}corr_matrix_{epsilon}.npy")
        weighted_global_edges = [(edge[0], edge[1], corr_matrix[edge[0]-1, edge[1]-1]) for edge in global_edges \
            if corr_matrix[edge[0]-1, edge[1]-1] > 0]
            
        division = hier_2D_SE_mini(weighted_global_edges, len(embeddings), n=n)
        print(datetime.now().strftime("%H:%M:%S"))

        prediction = decode(division)
        
        labels_true = df['event_id'].tolist()
        n_clusters = len(list(set(labels_true)))
        print('n_clusters gt: ', n_clusters)

        nmi, ami, ari = evaluate_labels(labels_true, prediction)
        print('n_clusters pred: ', len(division))
        print('nmi: ', nmi)
        print('ami: ', ami)
        print('ari: ', ari)
        
        with open(f"open_set_{epsilon}.txt", 'a') as f:
            f.write("block:" + str(block) + '\n')
            f.write("division:"+str(division)+ '\n')
            f.write('Runtime: ' + str(time.time() - start_time) + " Seconds" + '\n')
            f.write('n_clusters gt: '+ str(len(list(set(labels_true))))+ '\n')
            f.write('n_clusters pred: ' + str(len(division)) + '\n')
            f.write('epsilon: ' + str(epsilon) + '\n')
            f.write('n: ' + str(n) + '\n')
            f.write('Sensitivity: ' + str(Sensitivity) + '\n')
            f.write('nmi: ' + str(nmi) + '\n')
            f.write('ami: ' + str(ami) + '\n')
            f.write('ari: ' + str(ari) + '\n' + '\n')
            
    return

def run_hier_2D_SE_mini_closed_set(save_path, n=300, e_a=True, e_s=True, epsilon=None):
    
    save_path = save_path + 'closed_set/'
    # Load test set dataframe
    test_set_df_np_path = save_path + 'test_set.npy'
    test_df_np = np.load(test_set_df_np_path, allow_pickle=True)
    columns = ['tweet_id', 'text', 'event_id', 'words', 'filtered_words',
              'entities', 'user_id', 'created_at', 'urls', 'hashtags', 'user_mentions']
    test_df = pd.DataFrame(data=test_df_np, columns=columns)
    print("Dataframe loaded.")
    
    all_node_features = [[str(u)] + \
                        [str(each) for each in (um if isinstance(um, (list, tuple)) else [])] + \
                        [str(h).lower() if isinstance(h, str) else str(h) for h in (hs if isinstance(hs, (list, tuple)) else [])] + \
                        [str(e) for e in (e if isinstance(e, (list, tuple)) else [])] \
                        for u, um, hs, e in \
                        zip(test_df['user_id'], test_df['user_mentions'], test_df['hashtags'], test_df['entities'])]


    # Load embeddings
    with open(f'{save_path}/SBERT_embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)

    start_time = time.time()
    stable_points, Sensitivity = get_stable_point(save_path, if_updata=True, epsilon=epsilon)
    default_num_neighbors = stable_points['first']

    global_edges = get_global_edges(all_node_features, epsilon, save_path, default_num_neighbors, e_a=e_a, e_s=e_s)
    corr_matrix = np.load(f"{save_path}corr_matrix_{epsilon}.npy")
    weighted_global_edges = [(edge[0], edge[1], corr_matrix[edge[0]-1, edge[1]-1]) for edge in global_edges \
        if corr_matrix[edge[0]-1, edge[1]-1] > 0]

    division = hier_2D_SE_mini(weighted_global_edges, len(embeddings), n=n)
    prediction = decode(division)

    labels_true = test_df['event_id'].tolist()
    n_clusters = len(list(set(labels_true)))
    print('n_clusters gt: ', n_clusters)

    print('n_clusters pred: ', len(division))
    
        
    return labels_true, prediction

def create_process_open_set(epsilon):
    target = run_hier_2D_SE_mini_open_set
    kwargs = {
        "n": 100,
        "e_a": True,
        "e_s": True,
        "test_with_one_block": True,
        "epsilon": epsilon
    }
       
    p = multiprocessing.Process(target=target, kwargs=kwargs)
    p.start()
    return p

def create_process_closed_set(epsilon):
    target = run_hier_2D_SE_mini_closed_set
    n = 300

    kwargs = {
        "n": n,
        "e_a": True,
        "e_s": True,
        "epsilon": epsilon
    }
    
    p = multiprocessing.Process(target=target, kwargs=kwargs)
    p.start()
    return p

def run_processes(epsilons, dataset_name, mode='close'):
    if mode == 'open':
        processes = [create_process_open_set(dataset_name, epsilon) for epsilon in epsilons]
    else:
        processes = [create_process_closed_set(dataset_name, epsilon) for epsilon in epsilons]
    for process in processes:
        process.join()
    print("All processes have completed their tasks.")

def make_symmetric(matrix):
    return np.triu(matrix) + np.triu(matrix, 1).T

def search_stable_points(embeddings, epsilon, path, max_num_neighbors = 200):
    print("size_of_embeddings",len(embeddings))
    corr_matrix = np.corrcoef(embeddings)  
    np.fill_diagonal(corr_matrix, 0)

    print("epsilon=",epsilon)
    s = -1
    if epsilon != None:
        max_ = np.max(corr_matrix)
        min_ = np.min(corr_matrix)
        print("Local Sensitivity:",(max_- min_))
        # delta = 10e-6  
        delta = 1 / len(embeddings)**2  
        beta = epsilon / (2 * np.log(2/delta))
        S = np.exp(-beta) * (max_- min_) * 2
        print("Smooth Sensitivity:", S)
        if S < 2:
            s = S
        else:
            s = 2

        print("Sensitivity=",s)
        corr_matrix = [[i+np.random.laplace(loc=0, scale=s/epsilon) for i in corr_matrix_] for corr_matrix_ in corr_matrix]
        corr_matrix = np.array(corr_matrix)
        corr_matrix = make_symmetric(corr_matrix)

    np.fill_diagonal(corr_matrix, 0)
    print(f"{path}"+f'corr_matrix_{epsilon}.npy')
    np.save(f"{path}"+f'corr_matrix_{epsilon}.npy', corr_matrix)
    corr_matrix_sorted_indices = np.argsort(corr_matrix)
    
    all_1dSEs = []
    seg = None
    for i in range(max_num_neighbors):
        dst_ids = corr_matrix_sorted_indices[:, -(i+1)]
        knn_edges = [(s+1, d+1, corr_matrix[s, d]) \
            for s, d in enumerate(dst_ids) if corr_matrix[s, d] > 0] # (s+1, d+1): +1 as node indexing starts from 1 instead of 0
        if i == 0:
            g = nx.Graph()
            g.add_weighted_edges_from(knn_edges)
            seg = SE(g)
            all_1dSEs.append(seg.calc_1dSE())
        else:
            all_1dSEs.append(seg.update_1dSE(all_1dSEs[-1], knn_edges))
    
    #print('all_1dSEs: ', all_1dSEs)
    stable_indices = []
    for i in range(1, len(all_1dSEs) - 1):
        if all_1dSEs[i] < all_1dSEs[i - 1] and all_1dSEs[i] < all_1dSEs[i + 1]:
            stable_indices.append(i)
    if len(stable_indices) == 0:
        print('No stable points found after checking k = 1 to ', max_num_neighbors)
        return 0, 0, s
    else:
        stable_SEs = [all_1dSEs[index] for index in stable_indices]
        index = stable_indices[stable_SEs.index(min(stable_SEs))]
        print('stable_indices: ', stable_indices)
        print('stable_SEs: ', stable_SEs)
        print('First stable point: k = ', stable_indices[0]+1, ', corresponding 1dSE: ', stable_SEs[0]) # n_neighbors should be index + 1
        print('Global stable point within the searching range: k = ', index + 1, \
            ', corresponding 1dSE: ', all_1dSEs[index]) # n_neighbors should be index + 1

    return stable_indices[0]+1, index + 1, s # first stable point, global stable point

def get_graph_edges(attributes):
    attr_nodes_dict = {}
    for i, l in enumerate(attributes):
        for attr in l:
            if attr not in attr_nodes_dict:
                attr_nodes_dict[attr] = [i+1] # node indexing starts from 1
            else:
                attr_nodes_dict[attr].append(i+1)

    for attr in attr_nodes_dict.keys():
        attr_nodes_dict[attr].sort()

    graph_edges = []
    for l in attr_nodes_dict.values():
        graph_edges += list(combinations(l, 2))
    return list(set(graph_edges))

def get_knn_edges(epsilon, path, default_num_neighbors):
    # corr_matrix = np.corrcoef(embeddings)
    # np.fill_diagonal(corr_matrix, 0)
    corr_matrix = np.load(f"{path}"+f'corr_matrix_{epsilon}.npy')
    corr_matrix_sorted_indices = np.argsort(corr_matrix)
    knn_edges = []
    for i in range(default_num_neighbors):
        dst_ids = corr_matrix_sorted_indices[:, -(i+1)]
        knn_edges += [(s+1, d+1) if s < d else (d+1, s+1) \
            for s, d in enumerate(dst_ids) if corr_matrix[s, d] > 0] # (s+1, d+1): +1 as node indexing starts from 1 instead of 0
    return list(set(knn_edges))

def get_global_edges(attributes, epsilon, folder, default_num_neighbors, e_a = True, e_s = True):
    graph_edges, knn_edges = [], []
    if e_a == True:
        graph_edges = get_graph_edges(attributes)
    if e_s == True:
        knn_edges = get_knn_edges(epsilon, folder, default_num_neighbors)
    return list(set(knn_edges + graph_edges))

def get_subgraphs_edges(clusters, graph_splits, weighted_global_edges):
    """
    Get the edges of each subgraph.

    :param clusters: A list containing the current clusters, each cluster is a list of nodes of the original graph.
    :type clusters: list of list of nodes
    :param graph_splits: A list of (start_index, end_index) pairs, each (start_index, end_index) pair indicates a subset of clusters, which will serve as the nodes of a new subgraph.
    :type graph_splits: list of tuples (int, int)
    :param weighted_global_edges: A list of (start node, end node, edge weight) tuples, each tuple is an edge in the original graph.
    :type weighted_global_edges: list of tuples (node, node, float)

    :return: A list containing the edges of all subgraphs.
    :rtype: list of lists of tuples (node, node, float)
    """
    all_subgraphs_edges = []
    for split in graph_splits:
        subgraph_clusters = clusters[split[0]:split[1]]
        subgraph_nodes = list(chain(*subgraph_clusters))
        subgraph_edges = [edge for edge in weighted_global_edges if edge[0] in subgraph_nodes and edge[1] in subgraph_nodes]
        all_subgraphs_edges.append(subgraph_edges)
    return all_subgraphs_edges

def get_best_egde(adj_matrix_, subgraphs_, all_subgraphs):
    adj_matrix = adj_matrix_.copy()
    
    mask_nodes = list(set(all_subgraphs+subgraphs_))  
    if len(mask_nodes) >0:
        adj_matrix[mask_nodes, :] = 0
        adj_matrix[:, mask_nodes] = 0

    flat_index = np.argmax(adj_matrix)
    egde = np.unravel_index(flat_index, adj_matrix.shape)
    weight = adj_matrix[egde]
    if weight > 0:
        return list(egde), weight
    else:
        print("There is no egdes in current G")
        return -1, -1

def get_best_node(adj_matrix_, subgraphs_, all_subgraphs):
    adj_matrix = adj_matrix_.copy()

    mask_nodes = list(set(all_subgraphs+subgraphs_))  
    nodes_to_modify = np.array(mask_nodes)
    adj_matrix[np.ix_(nodes_to_modify, nodes_to_modify)] = 0

    distance = adj_matrix[subgraphs_].sum(axis=0)
    distance_sort_arg = np.argsort(distance)[::-1]
    distance_sort = np.sort(distance)[::-1]
    avg = np.mean(distance[distance>0])
    indices = distance_sort[distance_sort>avg]

    if len(indices) > 0:
        return distance_sort_arg[:len(indices)].tolist(), distance_sort[:len(indices)].tolist()
    else:
        print("There are no edges connected to the current subgraph")
        return -1, -1

def get_subgraphs(adj_matrix, division, n, k_max):
    merged_rows_matrix = np.vstack([ adj_matrix[np.array(ls_)-1].sum(axis=0).tolist() for ls_ in division ])
    final_sum = np.array([ merged_rows_matrix[:, np.array(ls_)-1].sum(axis=1).tolist() for ls_ in division ] )
    np.fill_diagonal(final_sum, 0)
    G = nx.from_numpy_array(final_sum)
    
    subgraphs = []
    all_subgraphs = [] 
    for k in range(k_max):
        subgraphs_ = []
        if len(final_sum) - len(all_subgraphs)<= n: 
            G.remove_nodes_from(all_subgraphs)
            subgraphs_ = list(G.nodes)
            subgraphs.append(subgraphs_)
            print(len(subgraphs_), subgraphs_)
            break

        max_edge_or_node, max_weight = get_best_egde(final_sum, subgraphs_, all_subgraphs)
        subgraphs_.extend(max_edge_or_node)
        all_subgraphs.extend(max_edge_or_node)
        while True:
            if len(subgraphs_) >= n:
                break
            node_, weight_ = get_best_node(final_sum, subgraphs_, all_subgraphs)
            if node_ == -1:
                max_edge_or_node, max_weight = get_best_egde(final_sum, subgraphs_, all_subgraphs)
                subgraphs_.extend(max_edge_or_node)
                all_subgraphs.extend(max_edge_or_node)
                continue
            else:
                if len(subgraphs_) + len(node_) > n:
                    index_ = n - len(subgraphs_)
                    subgraphs_.extend(node_[:index_])
                    all_subgraphs.extend(node_[:index_])
                else:
                    subgraphs_.extend(node_)
                    all_subgraphs.extend(node_)
        subgraphs.append(subgraphs_)
        # print(len(subgraphs_), subgraphs_)

    # subgraphs = [[element + 1 for element in row] for row in subgraphs]

    new_division = []
    for subgraphs_index in subgraphs:
        new_division_ = []
        for index in subgraphs_index:
            new_division_.append(division[index])
        new_division.append(new_division_)
        
    return new_division

def hier_2D_SE_mini(weighted_global_edges, n_messages, n = 100):
    '''
    hierarchical 2D SE minimization
    '''
    ite = 0
    # initially, each node (message) is in its own cluster
    # node encoding starts from 1

    G = nx.Graph()
    G.add_weighted_edges_from(weighted_global_edges)
    adj_matrix = nx.to_numpy_array(G)

    clusters = [[i] for i in list(G.nodes)]
    while True:
        ite += 1
        print('\n=========Iteration ', str(ite), '=========')
        n_clusters = len(clusters)
        graph_splits = [(s, min(s+n, n_clusters)) for s in range(0, n_clusters, n)] # [s, e)
        # all_subgraphs_edges = get_subgraphs_edges(clusters, graph_splits, weighted_global_edges)

        if 1:
            subgraphs = get_subgraphs(adj_matrix, clusters, n, len(graph_splits))

            all_subgraphs_edges = []
            for subgraph_nodes in subgraphs:
                subgraph_nodes = [str(item) for sublist in subgraph_nodes for item in sublist]
                subgraph_edges = [(int(edge[0]),int(edge[1]),edge[2]) for edge in weighted_global_edges 
                                  if str(edge[0]) in subgraph_nodes and str(edge[1]) in subgraph_nodes]
                all_subgraphs_edges.append(subgraph_edges)

        else:
            all_subgraphs_edges = get_subgraphs_edges(clusters, graph_splits, weighted_global_edges)


        last_clusters = clusters
        print(f"the number of clusters: {len(last_clusters)}")
        clusters = []
        for i, subgraph_edges in enumerate(all_subgraphs_edges):
            print('\tSubgraph ', str(i+1))

            g = nx.Graph()
            g.add_weighted_edges_from(subgraph_edges)
            seg = SE(g)
            if 1:
                seg.division = {j: cluster for j, cluster in enumerate(subgraphs[i]) }
                # print({j: cluster for j, cluster in enumerate(subgraphs[i]) })
            else:
                seg.division = {j: cluster for j, cluster in enumerate(last_clusters[graph_splits[i][0]:graph_splits[i][1]])}
                # print(seg.division)
            seg.add_isolates()
            for k in seg.division.keys():
                for node in seg.division[k]:
                    seg.graph.nodes[node]['comm'] = k
            seg.update_struc_data()
            seg.update_struc_data_2d()
            seg.update_division_MinSE()

            print(f"size of subgraph{str(i+1)}: {len(subgraphs[i])} to {len(list(seg.division.values()))}")

            clusters += list(seg.division.values())

        if len(graph_splits) == 1:
            break
        if clusters == last_clusters:
            n *= 2
    return clusters

class SE:
    def __init__(self, graph: nx.Graph):
        self.graph = graph.copy()
        self.vol = self.get_vol()
        self.division = {}  # {comm1: [node11, node12, ...], comm2: [node21, node22, ...], ...}
        self.struc_data = {}  # {comm1: [vol1, cut1, community_node_SE, leaf_nodes_SE], comm2:[vol2, cut2, community_node_SE, leaf_nodes_SE]，... }
        self.struc_data_2d = {} # {comm1: {comm2: [vol_after_merge, cut_after_merge, comm_node_SE_after_merge, leaf_nodes_SE_after_merge], comm3: [], ...}, ...}

    def get_vol(self):
        '''
        get the volume of the graph
        '''
        return cuts.volume(self.graph, self.graph.nodes, weight = 'weight')

    def calc_1dSE(self):
        '''
        get the 1D SE of the graph
        '''
        SE = 0
        for n in self.graph.nodes:
            d = cuts.volume(self.graph, [n], weight = 'weight')
            SE += - (d / self.vol) * math.log2(d / self.vol)
        return SE

    def update_1dSE(self, original_1dSE, new_edges):
        '''
        get the updated 1D SE after new edges are inserted into the graph
        '''
    
        affected_nodes = []
        for edge in new_edges:
            affected_nodes += [edge[0], edge[1]]
        affected_nodes = set(affected_nodes)

        original_vol = self.vol
        original_degree_dict = {node:0 for node in affected_nodes}
        for node in affected_nodes.intersection(set(self.graph.nodes)):
            original_degree_dict[node] = self.graph.degree(node, weight = 'weight')

        # insert new edges into the graph
        self.graph.add_weighted_edges_from(new_edges)

        self.vol = self.get_vol()
        updated_vol = self.vol
        updated_degree_dict = {}
        for node in affected_nodes:
            updated_degree_dict[node] = self.graph.degree(node, weight = 'weight')
        
        updated_1dSE = (original_vol / updated_vol) * (original_1dSE - math.log2(original_vol / updated_vol))
        for node in affected_nodes:
            d_original = original_degree_dict[node]
            d_updated = updated_degree_dict[node]
            if d_original != d_updated:
                if d_original != 0:
                    updated_1dSE += (d_original / updated_vol) * math.log2(d_original / updated_vol)
                updated_1dSE -= (d_updated / updated_vol) * math.log2(d_updated / updated_vol)

        return updated_1dSE

    def get_cut(self, comm):
        '''
        get the sum of the degrees of the cut edges of community comm
        '''
        return cuts.cut_size(self.graph, comm, weight = 'weight')

    def get_volume(self, comm):
        '''
        get the volume of community comm
        '''
        return cuts.volume(self.graph, comm, weight = 'weight')

    def calc_2dSE(self):
        '''
        get the 2D SE of the graph
        '''
        SE = 0
        for comm in self.division.values():
            g = self.get_cut(comm)
            v = self.get_volume(comm)
            SE += - (g / self.vol) * math.log2(v / self.vol)
            for node in comm:
                d = self.graph.degree(node, weight = 'weight')
                SE += - (d / self.vol) * math.log2(d / v)
        return SE

    def show_division(self):
        print(self.division)
        return self.division

    def show_struc_data(self):
        print(self.struc_data)
    
    def show_struc_data_2d(self):
        print(self.struc_data_2d)
        return self.struc_data_2d
        
    def print_graph(self):
        fig, ax = plt.subplots()
        nx.draw(self.graph, ax=ax, with_labels=True)
        plt.show()
    
    def update_struc_data(self):
        '''
        calculate the volume, cut, communitiy mode SE, and leaf nodes SE of each cummunity, 
        then store them into self.struc_data
        '''
        self.struc_data = {} # {comm1: [vol1, cut1, community_node_SE, leaf_nodes_SE], comm2:[vol2, cut2, community_node_SE, leaf_nodes_SE]，... }
        for vname in self.division.keys():
            comm = self.division[vname]
            volume = self.get_volume(comm)
            cut = self.get_cut(comm)
            if volume == 0:
                vSE = 0
            else:
                vSE = - (cut / self.vol) * math.log2(volume / self.vol)
            vnodeSE = 0
            for node in comm:
                d = self.graph.degree(node, weight = 'weight')
                if d != 0:
                    vnodeSE -= (d / self.vol) * math.log2(d / volume)
            self.struc_data[vname] = [volume, cut, vSE, vnodeSE]

    def update_struc_data_2d(self):
        '''
        calculate the volume, cut, communitiy mode SE, and leaf nodes SE after merging each pair of cummunities, 
        then store them into self.struc_data_2d
        '''
        self.struc_data_2d = {} # {(comm1, comm2): [vol_after_merge, cut_after_merge, comm_node_SE_after_merge, leaf_nodes_SE_after_merge], (comm1, comm3): [], ...}
        comm_num = len(self.division)
        for i in range(comm_num):
            for j in range(i + 1, comm_num):
                v1 = list(self.division.keys())[i]
                v2 = list(self.division.keys())[j]
                if v1 < v2:
                    k = (v1, v2)
                else:
                    k = (v2, v1)

                comm_merged = self.division[v1] + self.division[v2]
                gm = self.get_cut(comm_merged)
                vm = self.struc_data[v1][0] + self.struc_data[v2][0]
                if self.struc_data[v1][0] == 0 or self.struc_data[v2][0] == 0:
                    vmSE = self.struc_data[v1][2] + self.struc_data[v2][2]
                    vmnodeSE = self.struc_data[v1][3] + self.struc_data[v2][3]
                else:
                    vmSE = - (gm / self.vol) * math.log2(vm / self.vol)
                    vmnodeSE = self.struc_data[v1][3] - (self.struc_data[v1][0]/ self.vol) * math.log2(self.struc_data[v1][0] / vm) + \
                            self.struc_data[v2][3] - (self.struc_data[v2][0]/ self.vol) * math.log2(self.struc_data[v2][0] / vm)
                self.struc_data_2d[k] = [vm, gm, vmSE, vmnodeSE]

    def init_division(self):
        '''
        initialize self.division such that each node assigned to its own community
        '''
        self.division = {}
        for node in self.graph.nodes:
            new_comm = node
            self.division[new_comm] = [node]
            self.graph.nodes[node]['comm'] = new_comm

    def add_isolates(self):
        '''
        add any isolated nodes into graph
        '''
        all_nodes = list(chain(*list(self.division.values())))
        all_nodes.sort()
        edge_nodes = list(self.graph.nodes)
        edge_nodes.sort()
        if all_nodes != edge_nodes:
            for node in set(all_nodes)-set(edge_nodes):
                self.graph.add_node(node)

    def update_division_MinSE(self):
        '''
        greedily update the encoding tree to minimize 2D SE
        '''
        def Mg_operator(v1, v2):
            '''
            MERGE operator. It calculates the delta SE caused by mergeing communities v1 and v2, 
            without actually merging them, i.e., the encoding tree won't be changed
            '''
            v1SE = self.struc_data[v1][2] 
            v1nodeSE = self.struc_data[v1][3]

            v2SE = self.struc_data[v2][2]
            v2nodeSE = self.struc_data[v2][3]

            if v1 < v2:
                k = (v1, v2)
            else:
                k = (v2, v1)
            vm, gm, vmSE, vmnodeSE = self.struc_data_2d[k]
            delta_SE = vmSE + vmnodeSE - (v1SE + v1nodeSE + v2SE + v2nodeSE)
            return delta_SE

        # continue merging any two communities that can cause the largest decrease in SE, 
        # until the SE can't be further reduced
        while True: 
            comm_num = len(self.division)
            delta_SE = 99999
            vm1 = None
            vm2 = None
            for i in range(comm_num):
                for j in range(i + 1, comm_num):
                    v1 = list(self.division.keys())[i]
                    v2 = list(self.division.keys())[j]
                    new_delta_SE = Mg_operator(v1, v2)
                    if new_delta_SE < delta_SE:
                        delta_SE = new_delta_SE
                        vm1 = v1
                        vm2 = v2

            if delta_SE < 0:
                # Merge v2 into v1, and update the encoding tree accordingly
                for node in self.division[vm2]:
                    self.graph.nodes[node]['comm'] = vm1
                self.division[vm1] += self.division[vm2]
                self.division.pop(vm2)

                volume = self.struc_data[vm1][0] + self.struc_data[vm2][0]
                cut = self.get_cut(self.division[vm1])
                vmSE = - (cut / self.vol) * math.log2(volume / self.vol)
                vmnodeSE = self.struc_data[vm1][3] - (self.struc_data[vm1][0]/ self.vol) * math.log2(self.struc_data[vm1][0] / volume) + \
                        self.struc_data[vm2][3] - (self.struc_data[vm2][0]/ self.vol) * math.log2(self.struc_data[vm2][0] / volume)
                self.struc_data[vm1] = [volume, cut, vmSE, vmnodeSE]
                self.struc_data.pop(vm2)

                struc_data_2d_new = {}
                for k in self.struc_data_2d.keys():
                    if k[0] == vm2 or k[1] == vm2:
                        continue
                    elif k[0] == vm1 or k[1] == vm1:
                        v1 = k[0]
                        v2 = k[1]
                        comm_merged = self.division[v1] + self.division[v2]
                        gm = self.get_cut(comm_merged)
                        vm = self.struc_data[v1][0] + self.struc_data[v2][0]
                        if self.struc_data[v1][0] == 0 or self.struc_data[v2][0] == 0:
                            vmSE = self.struc_data[v1][2] + self.struc_data[v2][2]
                            vmnodeSE = self.struc_data[v1][3] + self.struc_data[v2][3]
                        else:
                            vmSE = - (gm / self.vol) * math.log2(vm / self.vol)
                            vmnodeSE = self.struc_data[v1][3] - (self.struc_data[v1][0]/ self.vol) * math.log2(self.struc_data[v1][0] / vm) + \
                                    self.struc_data[v2][3] - (self.struc_data[v2][0]/ self.vol) * math.log2(self.struc_data[v2][0] / vm)
                        struc_data_2d_new[k] = [vm, gm, vmSE, vmnodeSE]
                    else:
                        struc_data_2d_new[k] = self.struc_data_2d[k]
                self.struc_data_2d = struc_data_2d_new
            else:
                break

def vanilla_2D_SE_mini(weighted_edges):
    '''
    vanilla (greedy) 2D SE minimization
    '''
    g = nx.Graph()
    g.add_weighted_edges_from(weighted_edges)
    
    seg = SE(g)
    seg.init_division()
    #seg.show_division()
    SE1D = seg.calc_1dSE()

    seg.update_struc_data()
    #seg.show_struc_data()
    seg.update_struc_data_2d()
    #seg.show_struc_data_2d()
    initial_SE2D = seg.calc_2dSE()

    seg.update_division_MinSE()
    communities = seg.division
    minimized_SE2D = seg.calc_2dSE()

    return SE1D, initial_SE2D, minimized_SE2D, communities

def test_vanilla_2D_SE_mini():
    weighted_edges = [(1, 2, 2), (1, 3, 4)]

    g = nx.Graph()
    g.add_weighted_edges_from(weighted_edges)
    A = nx.adjacency_matrix(g).todense()
    print('adjacency matrix: \n', A)
    print('g.nodes: ', g.nodes)
    print('g.edges: ', g.edges)
    print('degrees of nodes: ', list(g.degree(g.nodes, weight = 'weight')))

    SE1D, initial_SE2D, minimized_SE2D, communities = vanilla_2D_SE_mini(weighted_edges)
    print('\n1D SE of the graph: ', SE1D)
    print('initial 2D SE of the graph: ', initial_SE2D)
    print('the minimum 2D SE of the graph: ', minimized_SE2D)
    print('communities detected: ', communities)
    return

def replaceAtUser(text):
    """ Replaces "@user" with "" """
    text = re.sub('@[^\s]+|RT @[^\s]+','',text)
    return text

def removeUnicode(text):
    """ Removes unicode strings like "\u002c" and "x96" """
    text = re.sub(r'(\\u[0-9A-Fa-f]+)',r'', text)       
    text = re.sub(r'[^\x00-\x7f]',r'',text)
    return text

def replaceURL(text):
    """ Replaces url address with "url" """
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','url',text)
    text = re.sub(r'#([^\s]+)', r'\1', text)
    return text

def replaceMultiExclamationMark(text):
    """ Replaces repetitions of exlamation marks """
    text = re.sub(r"(\!)\1+", '!', text)
    return text

def replaceMultiQuestionMark(text):
    """ Replaces repetitions of question marks """
    text = re.sub(r"(\?)\1+", '?', text)
    return text

def removeEmoticons(text):
    """ Removes emoticons from text """
    text = re.sub(':\)|;\)|:-\)|\(-:|:-D|=D|:P|xD|X-p|\^\^|:-*|\^\.\^|\^\-\^|\^\_\^|\,-\)|\)-:|:\'\(|:\(|:-\(|:\S|T\.T|\.\_\.|:<|:-\S|:-<|\*\-\*|:O|=O|=\-O|O\.o|XO|O\_O|:-\@|=/|:/|X\-\(|>\.<|>=\(|D:', '', text)
    return text

def removeNewLines(text):
    text = re.sub('\n', '', text)
    return text

def preprocess_sentence(s):
    return removeNewLines(replaceAtUser(removeEmoticons(replaceMultiQuestionMark(replaceMultiExclamationMark(removeUnicode(replaceURL(s)))))))

def preprocess_french_sentence(s):
    return removeNewLines(replaceAtUser(removeEmoticons(replaceMultiQuestionMark(replaceMultiExclamationMark(replaceURL(s))))))

def SBERT_embed(s_list, language):
    '''
    Use Sentence-BERT to embed sentences.
    s_list: a list of sentences/ tokens to be embedded.
    language: the language of the sentences ('English', 'French', 'Arabic').
    output: the embeddings of the sentences/ tokens.
    '''
    # Model paths or names for each language
    model_map = {
        'English': '../model/model_needed/all-MiniLM-L6-v2',
        'French': '../model/model_needed/distiluse-base-multilingual-cased-v1',
        'Arabic': '../model/model_needed/paraphrase-multilingual-mpnet-base-v2'
    }

    # Default model for Hugging Face
    hf_model_map = {
        'English': 'sentence-transformers/all-MiniLM-L6-v2',
        'French': 'sentence-transformers/distiluse-base-multilingual-cased-v1',
        'Arabic': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    }

    # Print language and model being used
    print(f"Embedding sentences in language: {language}")
    
    # Determine model path
    model_path = model_map.get(language)
    if not model_path:
        raise ValueError(f"Unsupported language: {language}. Supported languages are: {', '.join(model_map.keys())}")

    print(f"Using model: {model_path}")

    # Load the model, downloading if necessary
    try:
        model = SentenceTransformer(model_path)
        print(f"Successfully loaded model from local path: {model_path}")
    except Exception as e:
        print(f"Model {model_path} not found locally. Attempting to download from Hugging Face...")
        model = SentenceTransformer(hf_model_map[language])
        print(f"Model downloaded from Hugging Face: {hf_model_map[language]}")

    # Compute embeddings
    embeddings = model.encode(s_list, convert_to_tensor=True, normalize_embeddings=True)
    print(f"Computed embeddings for {len(s_list)} sentences/tokens.")
    
    return embeddings.cpu()

def evaluate_labels(labels_true, labels_pred):
    nmi = metrics.normalized_mutual_info_score(labels_true, labels_pred)
    ami = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
    ari = metrics.adjusted_rand_score(labels_true, labels_pred)
    return nmi, ami, ari

def decode(division):
    if type(division) is dict:
        prediction_dict = {m: event for event, messages in division.items() for m in messages}
    elif type(division) is list:
        prediction_dict = {m: event for event, messages in enumerate(division) for m in messages}
    prediction_dict_sorted = dict(sorted(prediction_dict.items()))
    return list(prediction_dict_sorted.values())

if __name__ == "__main__":
    from dataset.dataloader import Event2012

    event2012 = Event2012()
    adpsemevent = ADPSEMEvent(event2012)
    adpsemevent.preprocess()
    predictions, ground_truths = adpsemevent.detection()
    adpsemevent.evaluate(predictions, ground_truths)

