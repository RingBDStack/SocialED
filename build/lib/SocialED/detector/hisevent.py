import os
import numpy as np
import pandas as pd
from os.path import exists
import pickle
import torch
from sentence_transformers import SentenceTransformer
import re
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score
from sklearn import metrics
from itertools import combinations, chain
import networkx as nx
from datetime import datetime
import math
from networkx.algorithms import cuts
from sklearn.model_selection import train_test_split
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.dataloader import DatasetLoader, DatasetLoader_mini




#2012 user_id,user_mentions,hashtags,entities,text,event_id
class HISEvent():
    def __init__(self, dataset, language):
        self.dataset = dataset
        self.language = language
        self.save_path = "../model/model_saved/hisevent/Event2012_mini/"

    def preprocess(self):
        preprocessor = Preprocessor(self.dataset)
        preprocessor.preprocess_data(self.save_path, self.dataset, self.language)

    def detection(self):
        ground_truths, predictions = run_hier_2D_SE_mini_data(self.save_path, n=300, e_a=True, e_s=True)
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
    def __init__(self, dataset):
        pass

    def get_data_test_df(self, save_path, dataset):
        df = dataset
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        test_set_df_np_path = save_path + 'test_set.npy'

        # 1. 检查 test_set.npy 是否已经存在，如果已存在，检查其是否符合预期
        if os.path.exists(test_set_df_np_path):
            print(f"File {test_set_df_np_path} already exists.")
            # 尝试加载现有的 test_set.npy 文件并检查其内容
            try:
                test_set = np.load(test_set_df_np_path)
                print(f"Loaded test set from {test_set_df_np_path}.")
                print(f"Shape of the array: {test_set.shape}")
                print(f"Data type of the array: {test_set.dtype}")
                print(f"First few rows:\n{test_set[:5]}")  # 打印前 5 行数据
            except Exception as e:
                print(f"Error loading {test_set_df_np_path}: {e}")
                print("The file might be corrupted or empty. Re-generating the file.")
        else:
            print(f"File {test_set_df_np_path} does not exist. Generating the test set...")

            # 2. 加载 test_mask 文件
            try:
                test_mask = torch.load(save_path + 'masks/test_mask.pt').cpu().detach().numpy()
                print("Test mask loaded successfully.")
            except Exception as e:
                print(f"Error loading test_mask.pt: {e}")
                return

            # 3. 检查 test_mask 是否有效，确保掩码的尺寸和内容是合理的
            print(f"Total samples in test mask: {len(test_mask)}")
            print(f"Number of True values in test mask: {np.sum(test_mask)}")
            if np.sum(test_mask) == 0:
                print("Warning: No True values in test mask! No test samples selected.")
                return
            if np.sum(test_mask) == len(test_mask):
                print("Warning: All values in test mask are True! This might not be correct.")
            
            # 4. 根据 test_mask 筛选数据
            test_mask = list(np.where(test_mask == True)[0])  # 获取所有 True 的索引
            test_df = df.iloc[test_mask]
            print(f"Number of samples in the test set: {len(test_df)}")
            if len(test_df) == 0:
                print("Error: The test set is empty after applying the mask.")
                return

            # 5. 将筛选后的 test_df 转换为 NumPy 数组
            test_df_np = test_df.to_numpy()

            # 6. 检查转换后的 test_df 是否为空或维度不匹配
            if test_df_np.shape[0] == 0:
                print("Error: Converted test set is empty.")
                return
            if test_df_np.shape[1] != df.shape[1]:
                print(f"Error: Column mismatch between original data and test set. Original columns: {df.shape[1]}, Test set columns: {test_df_np.shape[1]}")
                return

            # 7. 保存 test_set.npy
            try:
                np.save(test_set_df_np_path, test_df_np)
                print(f"Test set saved successfully to {test_set_df_np_path}.")
            except Exception as e:
                print(f"Error saving {test_set_df_np_path}: {e}")

            # 8. 重新加载并检查保存的文件
            try:
                test_set = np.load(test_set_df_np_path)
                print(f"Loaded saved test set from {test_set_df_np_path}.")
                print(f"Shape of the array: {test_set.shape}")
                print(f"Data type of the array: {test_set.dtype}")
                print(f"First few rows:\n{test_set[:5]}")  # 打印前 5 行数据
            except Exception as e:
                print(f"Error loading {test_set_df_np_path}: {e}")
                print("The file might be corrupted or empty.")
                return

        return

    def get_data_messages_embeddings(self, save_path, language):

        SBERT_embedding_path = f'{save_path}/SBERT_embeddings.pkl'
        if not exists(SBERT_embedding_path):
            test_set_df_np_path = save_path + 'test_set.npy'
            test_df_np = np.load(test_set_df_np_path, allow_pickle=True)
            test_df = pd.DataFrame(data=test_df_np,
                                   columns=['tweet_id', 'text', 'event_id', 'words', 'filtered_words',
                                        'entities', 'user_id', 'created_at', 'urls', 'hashtags', 'user_mentions'
                                        ])
            print("Dataframe loaded.")
            # print('test_df.head(5): \n', test_df.head(5))
            # print('test_df.head(5)[\'text\']: \n', test_df.head(5)['text'])

            processed_text = [preprocess_sentence(s) for s in
                              test_df['text'].values]  # hastags are kept (with '#' removed). RTs are removed.
            print('message text contents preprocessed.')

            # get SBERT embeddings
            embeddings = SBERT_embed(processed_text, language)

            # store SBERT embeddings
            with open(SBERT_embedding_path, 'wb') as fp:
                pickle.dump(embeddings, fp)
            print('SBERT embeddings stored.')
        return


    def split_and_save_masks(self, df, save_dir, train_size=0.7, val_size=0.1, test_size=0.2, random_seed=42):
        """
        Splits the DataFrame into training, validation, and test sets, and saves the indices (masks) as .pt files.
        
        Parameters:
        - df (pd.DataFrame): The DataFrame to be split.
        - train_size (float): Proportion of data to be used for training (default is 70%).
        - val_size (float): Proportion of data to be used for validation (default is 10%).
        - test_size (float): Proportion of data to be used for testing (default is 20%).
        - random_seed (int): Seed for reproducibility (default is 42).
        - save_dir (str): Directory where the masks will be saved.
        
        Returns:
        - None
        """
        

        # 1. Check if the sizes add up to 1
        if train_size + val_size + test_size != 1.0:
            raise ValueError("train_size + val_size + test_size must equal 1.0")
        
        # 2. Ensure the dataframe is not empty
        if df.empty:
            raise ValueError("The input DataFrame is empty.")
        
        print(f"Total samples in DataFrame: {len(df)}")

        # 3. Set random seed for reproducibility
        torch.manual_seed(random_seed)
        
        # 4. Split data into train + temp (for validation and test)
        print("Splitting data into train and temp...")
        train_data, temp_data = train_test_split(df, train_size=train_size, random_state=random_seed)
        
        # 5. Split the temp data into validation and test
        print("Splitting temp data into validation and test...")
        val_data, test_data = train_test_split(temp_data, train_size=val_size / (val_size + test_size), random_state=random_seed)
        
        # 6. Create index masks for each split
        train_mask = train_data.index.tolist()
        val_mask = val_data.index.tolist()
        test_mask = test_data.index.tolist()

        # 7. Create a boolean mask (True for the data points in the respective set, False otherwise)
        # Initialize all as False
        full_train_mask = torch.zeros(len(df), dtype=torch.bool)
        full_val_mask = torch.zeros(len(df), dtype=torch.bool)
        full_test_mask = torch.zeros(len(df), dtype=torch.bool)

        # Mark the corresponding indices as True
        full_train_mask[train_mask] = True
        full_val_mask[val_mask] = True
        full_test_mask[test_mask] = True

        # Print the sizes of each split for verification
        print(f"Training data size: {full_train_mask.sum()} samples")
        print(f"Validation data size: {full_val_mask.sum()} samples")
        print(f"Test data size: {full_test_mask.sum()} samples")

        # 8. Create directory to save masks if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # 9. Define the mask file paths
        mask_paths = {
            'train_mask.pt': full_train_mask,
            'val_mask.pt': full_val_mask,
            'test_mask.pt': full_test_mask
        }

        # 10. Save the masks if the files do not exist
        for filename, mask in mask_paths.items():
            mask_path = os.path.join(save_dir, filename)
            
            if not os.path.exists(mask_path):
                try:
                    # Save the mask as a tensor
                    torch.save(mask, mask_path)
                    print(f"Saved {filename} at {mask_path}")
                except Exception as e:
                    print(f"Error saving {filename} at {mask_path}: {e}")
            else:
                print(f"{filename} already exists at {mask_path}")
            
            # 11. After saving, verify if the file is non-empty
            if os.path.exists(mask_path):
                saved_mask = torch.load(mask_path)
                if saved_mask.numel() == 0:
                    print(f"Warning: {filename} is empty.")
                else:
                    print(f"Successfully saved and verified {filename} with {saved_mask.numel()} elements.")

        print("Mask generation and saving process completed.")

    # Example usage
    # df = pd.DataFrame({...})  # Assuming df is an existing DataFrame
    # split_and_save_masks(df)


    def preprocess_data(self, save_path, dataset, language):
        # load raw data

        print("Data converted to dataframe.")

        save_dir = os.path.join(save_path, 'masks')
        self.split_and_save_masks(dataset, save_dir)
        # get test set df
        self.get_data_test_df(save_path, dataset)

        # get SBERT embeddings
        self.get_data_messages_embeddings(save_path, language)

        return

def get_stable_point(path):
    stable_point_path = path + 'stable_point.pkl'
    if not exists(stable_point_path):
        embeddings_path = path + 'SBERT_embeddings.pkl'
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"Loaded embeddings: {embeddings}")
        print(f"Shape of embeddings: {np.shape(embeddings)}")

        first_stable_point, global_stable_point = search_stable_points(embeddings)
        stable_points = {'first': first_stable_point, 'global': global_stable_point}
        with open(stable_point_path, 'wb') as fp:
            pickle.dump(stable_points, fp)
        print('stable points stored.')

    with open(stable_point_path, 'rb') as f:
        stable_points = pickle.load(f)
    print('stable points loaded.')
    return stable_points

def run_hier_2D_SE_mini_data(save_path, n=300, e_a=True, e_s=True):

    # load test_set_df
    test_set_df_np_path = save_path + 'test_set.npy'
    test_df_np = np.load(test_set_df_np_path, allow_pickle=True)
    test_df = pd.DataFrame(data=test_df_np,
                           columns=['tweet_id', 'text', 'event_id', 'words', 'filtered_words',
                                        'entities', 'user_id', 'created_at', 'urls', 'hashtags', 'user_mentions'
                                        ])
    print("Dataframe loaded.")
    all_node_features = [[str(u)] + \
                         [str(each) for each in um] + \
                         [h.lower() for h in hs] + \
                         e \
                         for u, um, hs, e in \
                         zip(test_df['user_id'], test_df['user_mentions'], test_df['hashtags'], test_df['entities'])]

    # load embeddings of the test set messages
    with open(f'{save_path}/SBERT_embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)

    stable_points = get_stable_point(save_path)
    default_num_neighbors = stable_points['first']

    global_edges = get_global_edges(all_node_features, embeddings, default_num_neighbors, e_a=e_a, e_s=e_s)
    corr_matrix = np.corrcoef(embeddings)

    np.fill_diagonal(corr_matrix, 0)
    weighted_global_edges = [(edge[0], edge[1], corr_matrix[edge[0] - 1, edge[1] - 1]) for edge in global_edges \
                             if corr_matrix[edge[0] - 1, edge[1] - 1] > 0]  # node encoding starts from 1

    division = hier_2D_SE_mini(weighted_global_edges, len(embeddings), n=n)
    prediction = decode(division)

    labels_true = test_df['event_id'].tolist()
    n_clusters = len(list(set(labels_true)))
    print('n_clusters gt: ', n_clusters)

    return labels_true, prediction




def search_stable_points(embeddings, max_num_neighbors=50):
    corr_matrix = np.corrcoef(embeddings)

    np.fill_diagonal(corr_matrix, 0)
    corr_matrix_sorted_indices = np.argsort(corr_matrix)

    all_1dSEs = []
    seg = None
    for i in range(max_num_neighbors):
        dst_ids = corr_matrix_sorted_indices[:, -(i + 1)]
        knn_edges = [(s + 1, d + 1, corr_matrix[s, d]) \
                     for s, d in enumerate(dst_ids) if
                     corr_matrix[s, d] > 0]  # (s+1, d+1): +1 as node indexing starts from 1 instead of 0
        if i == 0:
            g = nx.Graph()
            g.add_weighted_edges_from(knn_edges)
            seg = SE(g)
            all_1dSEs.append(seg.calc_1dSE())
        else:
            all_1dSEs.append(seg.update_1dSE(all_1dSEs[-1], knn_edges))

    # print('all_1dSEs: ', all_1dSEs)
    stable_indices = []
    for i in range(1, len(all_1dSEs) - 1):
        if all_1dSEs[i] < all_1dSEs[i - 1] and all_1dSEs[i] < all_1dSEs[i + 1]:
            stable_indices.append(i)
    if len(stable_indices) == 0:
        print('No stable points found after checking k = 1 to ', max_num_neighbors)
        return 0, 0
    else:
        stable_SEs = [all_1dSEs[index] for index in stable_indices]
        index = stable_indices[stable_SEs.index(min(stable_SEs))]
        print('stable_indices: ', stable_indices)
        print('stable_SEs: ', stable_SEs)
        print('First stable point: k = ', stable_indices[0] + 1, ', correspoding 1dSE: ',
              stable_SEs[0])  # n_neighbors should be index + 1
        print('Global stable point within the searching range: k = ', index + 1, \
              ', correspoding 1dSE: ', all_1dSEs[index])  # n_neighbors should be index + 1
    return stable_indices[0] + 1, index + 1  # first stable point, global stable point

def get_graph_edges(attributes):
    attr_nodes_dict = {}
    for i, l in enumerate(attributes):
        for attr in l:
            if attr not in attr_nodes_dict:
                attr_nodes_dict[attr] = [i + 1]  # node indexing starts from 1
            else:
                attr_nodes_dict[attr].append(i + 1)

    for attr in attr_nodes_dict.keys():
        attr_nodes_dict[attr].sort()

    graph_edges = []
    for l in attr_nodes_dict.values():
        graph_edges += list(combinations(l, 2))
    return list(set(graph_edges))

def get_knn_edges(embeddings, default_num_neighbors):
    corr_matrix = np.corrcoef(embeddings)
    np.fill_diagonal(corr_matrix, 0)
    corr_matrix_sorted_indices = np.argsort(corr_matrix)
    knn_edges = []
    for i in range(default_num_neighbors):
        dst_ids = corr_matrix_sorted_indices[:, -(i + 1)]
        knn_edges += [(s + 1, d + 1) if s < d else (d + 1, s + 1) \
                      for s, d in enumerate(dst_ids) if
                      corr_matrix[s, d] > 0]  # (s+1, d+1): +1 as node indexing starts from 1 instead of 0
    return list(set(knn_edges))

def get_global_edges(attributes, embeddings, default_num_neighbors, e_a=True, e_s=True):
    graph_edges, knn_edges = [], []
    if e_a == True:
        graph_edges = get_graph_edges(attributes)
    if e_s == True:
        knn_edges = get_knn_edges(embeddings, default_num_neighbors)
    return list(set(knn_edges + graph_edges))

def get_subgraphs_edges(clusters, graph_splits, weighted_global_edges):
    '''
    get the edges of each subgraph

    clusters: a list containing the current clusters, each cluster is a list of nodes of the original graph
    graph_splits: a list of (start_index, end_index) pairs, each (start_index, end_index) pair indicates a subset of clusters, 
        which will serve as the nodes of a new subgraph
    weighted_global_edges: a list of (start node, end node, edge weight) tuples, each tuple is an edge in the original graph

    return: all_subgraphs_edges: a list containing the edges of all subgraphs
    '''
    all_subgraphs_edges = []
    for split in graph_splits:
        subgraph_clusters = clusters[split[0]:split[1]]
        subgraph_nodes = list(chain(*subgraph_clusters))
        subgraph_edges = [edge for edge in weighted_global_edges if
                          edge[0] in subgraph_nodes and edge[1] in subgraph_nodes]
        all_subgraphs_edges.append(subgraph_edges)
    return all_subgraphs_edges

def hier_2D_SE_mini(weighted_global_edges, n_messages, n=100):
    '''
    hierarchical 2D SE minimization
    '''
    ite = 1
    # initially, each node (message) is in its own cluster
    # node encoding starts from 1
    clusters = [[i + 1] for i in range(n_messages)]
    while True:
        print('\n=========Iteration ', str(ite), '=========')
        n_clusters = len(clusters)
        graph_splits = [(s, min(s + n, n_clusters)) for s in range(0, n_clusters, n)]  # [s, e)
        all_subgraphs_edges = get_subgraphs_edges(clusters, graph_splits, weighted_global_edges)
        last_clusters = clusters
        clusters = []
        print('111111111111111111111111111Number of subgraphs: ', len(graph_splits))
        for i, subgraph_edges in enumerate(all_subgraphs_edges):
            print('\tSubgraph ', str(i + 1))
            g = nx.Graph()
            g.add_weighted_edges_from(subgraph_edges)
            seg = SE(g)
            seg.division = {j: cluster for j, cluster in
                            enumerate(last_clusters[graph_splits[i][0]:graph_splits[i][1]])}
            seg.add_isolates()
            for k in seg.division.keys():
                for node in seg.division[k]:
                    seg.graph.nodes[node]['comm'] = k
            seg.update_struc_data()
            seg.update_struc_data_2d()
            seg.update_division_MinSE()

            clusters += list(seg.division.values())
        if len(graph_splits) == 1:
            break
        if clusters == last_clusters:
            n *= 2
    return clusters

# SE
class SE:
    def __init__(self, graph: nx.Graph):
        self.graph = graph.copy()
        self.vol = self.get_vol()
        self.division = {}  # {comm1: [node11, node12, ...], comm2: [node21, node22, ...], ...}
        self.struc_data = {}  # {comm1: [vol1, cut1, community_node_SE, leaf_nodes_SE], comm2:[vol2, cut2, community_node_SE, leaf_nodes_SE]，... }
        self.struc_data_2d = {}  # {comm1: {comm2: [vol_after_merge, cut_after_merge, comm_node_SE_after_merge, leaf_nodes_SE_after_merge], comm3: [], ...}, ...}

    def get_vol(self):
        '''
        get the volume of the graph
        '''
        return cuts.volume(self.graph, self.graph.nodes, weight='weight')

    def calc_1dSE(self):
        '''
        get the 1D SE of the graph
        '''
        SE = 0
        for n in self.graph.nodes:
            d = cuts.volume(self.graph, [n], weight='weight')
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
        original_degree_dict = {node: 0 for node in affected_nodes}
        for node in affected_nodes.intersection(set(self.graph.nodes)):
            original_degree_dict[node] = self.graph.degree(node, weight='weight')

        # insert new edges into the graph
        self.graph.add_weighted_edges_from(new_edges)

        self.vol = self.get_vol()
        updated_vol = self.vol
        updated_degree_dict = {}
        for node in affected_nodes:
            updated_degree_dict[node] = self.graph.degree(node, weight='weight')

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
        return cuts.cut_size(self.graph, comm, weight='weight')

    def get_volume(self, comm):
        '''
        get the volume of community comm
        '''
        return cuts.volume(self.graph, comm, weight='weight')

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
                d = self.graph.degree(node, weight='weight')
                SE += - (d / self.vol) * math.log2(d / v)
        return SE

    def show_division(self):
        print(self.division)

    def show_struc_data(self):
        print(self.struc_data)

    def show_struc_data_2d(self):
        print(self.struc_data_2d)

    def print_graph(self):
        fig, ax = plt.subplots()
        nx.draw(self.graph, ax=ax, with_labels=True)
        plt.show()

    def update_struc_data(self):
        '''
        calculate the volume, cut, communitiy mode SE, and leaf nodes SE of each cummunity, 
        then store them into self.struc_data
        '''
        self.struc_data = {}  # {comm1: [vol1, cut1, community_node_SE, leaf_nodes_SE], comm2:[vol2, cut2, community_node_SE, leaf_nodes_SE]，... }
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
                d = self.graph.degree(node, weight='weight')
                if d != 0:
                    vnodeSE -= (d / self.vol) * math.log2(d / volume)
            self.struc_data[vname] = [volume, cut, vSE, vnodeSE]

    def update_struc_data_2d(self):
        '''
        calculate the volume, cut, communitiy mode SE, and leaf nodes SE after merging each pair of cummunities, 
        then store them into self.struc_data_2d
        '''
        self.struc_data_2d = {}  # {(comm1, comm2): [vol_after_merge, cut_after_merge, comm_node_SE_after_merge, leaf_nodes_SE_after_merge], (comm1, comm3): [], ...}
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
                    vmnodeSE = self.struc_data[v1][3] - (self.struc_data[v1][0] / self.vol) * math.log2(
                        self.struc_data[v1][0] / vm) + \
                               self.struc_data[v2][3] - (self.struc_data[v2][0] / self.vol) * math.log2(
                        self.struc_data[v2][0] / vm)
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
            for node in set(all_nodes) - set(edge_nodes):
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
                vmnodeSE = self.struc_data[vm1][3] - (self.struc_data[vm1][0] / self.vol) * math.log2(
                    self.struc_data[vm1][0] / volume) + \
                           self.struc_data[vm2][3] - (self.struc_data[vm2][0] / self.vol) * math.log2(
                    self.struc_data[vm2][0] / volume)
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
                            vmnodeSE = self.struc_data[v1][3] - (self.struc_data[v1][0] / self.vol) * math.log2(
                                self.struc_data[v1][0] / vm) + \
                                       self.struc_data[v2][3] - (self.struc_data[v2][0] / self.vol) * math.log2(
                                self.struc_data[v2][0] / vm)
                        struc_data_2d_new[k] = [vm, gm, vmSE, vmnodeSE]
                    else:
                        struc_data_2d_new[k] = self.struc_data_2d[k]
                self.struc_data_2d = struc_data_2d_new
            else:
                break

def vanilla_2D_SE_mini(weighted_edges):
    """
    vanilla (greedy) 2D SE minimization
    """
    g = nx.Graph()
    g.add_weighted_edges_from(weighted_edges)

    seg = SE(g)
    seg.init_division()
    # seg.show_division()
    SE1D = seg.calc_1dSE()

    seg.update_struc_data()
    # seg.show_struc_data()
    seg.update_struc_data_2d()
    # seg.show_struc_data_2d()
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
    print('degrees of nodes: ', list(g.degree(g.nodes, weight='weight')))

    SE1D, initial_SE2D, minimized_SE2D, communities = vanilla_2D_SE_mini(weighted_edges)
    print('\n1D SE of the graph: ', SE1D)
    print('initial 2D SE of the graph: ', initial_SE2D)
    print('the minimum 2D SE of the graph: ', minimized_SE2D)
    print('communities detected: ', communities)
    return

# utils

def replaceAtUser(text):
    """ Replaces "@user" with "" """
    text = re.sub('@[^\s]+|RT @[^\s]+', '', text)
    return text


def removeUnicode(text):
    """ Removes unicode strings like "\u002c" and "x96" """
    text = re.sub(r'(\\u[0-9A-Fa-f]+)', r'', text)
    text = re.sub(r'[^\x00-\x7f]', r'', text)
    return text


def replaceURL(text):
    """ Replaces url address with "url" """
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'url', text)
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
    text = re.sub(
        ':\)|;\)|:-\)|\(-:|:-D|=D|:P|xD|X-p|\^\^|:-*|\^\.\^|\^\-\^|\^\_\^|\,-\)|\)-:|:\'\(|:\(|:-\(|:\S|T\.T|\.\_\.|:<|:-\S|:-<|\*\-\*|:O|=O|=\-O|O\.o|XO|O\_O|:-\@|=/|:/|X\-\(|>\.<|>=\(|D:',
        '', text)
    return text


def removeNewLines(text):
    text = re.sub('\n', '', text)
    return text


def preprocess_sentence(s):
    return removeNewLines(replaceAtUser(
        removeEmoticons(replaceMultiQuestionMark(replaceMultiExclamationMark(removeUnicode(replaceURL(s)))))))


def preprocess_french_sentence(s):
    return removeNewLines(
        replaceAtUser(removeEmoticons(replaceMultiQuestionMark(replaceMultiExclamationMark(replaceURL(s))))))


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



def evaluate(labels_true, labels_pred):
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    ami = adjusted_mutual_info_score(labels_true, labels_pred)
    ari = adjusted_rand_score(labels_true, labels_pred)
    return nmi, ami, ari


def decode(division):
    if type(division) is dict:
        prediction_dict = {m: event for event, messages in division.items() for m in messages}
    elif type(division) is list:
        prediction_dict = {m: event for event, messages in enumerate(division) for m in messages}
    prediction_dict_sorted = dict(sorted(prediction_dict.items()))
    return list(prediction_dict_sorted.values())


if __name__ == "__main__":

    #Data = DatasetLoader("MAVEN")
    Data = DatasetLoader_mini("Event2012")

    dataset = Data.load_data()
    language = Data.get_dataset_language()
    hisevent = HISEvent(dataset, language)
    hisevent.preprocess()
    predictions, ground_truths = hisevent.detection()
    hisevent.evaluate(predictions, ground_truths)


