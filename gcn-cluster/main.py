# Utility packages
import time
import numpy as np
import pandas as pd
from utils import export_to_excel, dic_to_csv
from networks import load_network, check_network

# Torch packages
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import SGConv
from torch_geometric.nn import GCNConv

# PIP packages
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from scipy.optimize import linear_sum_assignment as linear_assignment

# 
import math

class GCN_Clustering():

  def __init__(
    self,
    train_mask: list,
    test_mask: list,
    val_mask: list,
    class_size: int,
    k:int,
    metric:str = 'euclidean',
    alpha:float = 0.95,
    linkage:str = 'ward',
    network: str = 'gcn'
    ):

    # Confirm network availability
    check_network(network)
    
    # Load parameters
    self.train_mask = train_mask
    self.test_mask = test_mask
    self.val_mask = val_mask
    self.class_size = class_size
    self.k = k
    self.metric = metric
    self.alpha = alpha
    self.linkage = linkage
    self.edge_index = None
    self.network = network
  
  def run( self, features: np.array, labels: np.array, ranked_list_path: str, num_classes: int):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Edge index
    edge_index = self.create_graph(ranked_list_path=ranked_list_path, features=features, labels=labels)
    edge_index = torch.tensor(edge_index)
    edge_index = edge_index.t().contiguous().to(device)

    # Data definition
    data = self.prepare_data(self.features, self.labels, edge_index, device)

    # Variables
    pNNeurons = 32
    pNEpochs = 400
    pLR = 0.001
    pNFeatures = len(self.features[0])

    # Initializing model
    model, optimizer = self.initialize_model(pNFeatures, pNNeurons, num_classes, device)

    # Training
    self.train_model(data, model, optimizer, pNEpochs, pLR)
    
    # Evaluating
    acc = self.evaluate_model(data, model)

    return acc

  def prepare_data(self, features, labels, edge_index, device):
    x = torch.tensor(features).to(device)
    y = torch.tensor(labels, dtype=torch.long).to(device)

    train_mask = torch.tensor(self.train_mask).to(device)
    test_mask = torch.tensor(self.test_mask).to(device)
    val_mask = torch.tensor(self.val_mask).to(device)

    data = Data(x=x.float(), edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)

    return data

  def initialize_model(self, pNFeatures, pNNeurons, num_classes, device):
    model = load_network(pNFeatures, pNNeurons, num_classes, self.network).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    
    return model, optimizer

  def train_model(self, data, model, optimizer, pNEpochs, pLR):
    model.train()
    for epoch in range(pNEpochs):
      # Divide pLR by 2 every 100 epochs
      if (epoch + 1) % 100 == 0:
        pLR /= 2

      optimizer.zero_grad()
      out = model(data)

      if self.check_overfit(data, out):
        print(f"Early stopping on epoch {epoch}")
        break

      loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
      loss.backward()
      optimizer.step()

  def check_overfit(self, data, out):
    _, pred = out.max(dim=1)
    correct = float(pred[data.train_mask].eq(data.y[data.train_mask]).sum().item())
    acc = correct / data.train_mask.sum().item()

    return acc == 1.0
  
  def evaluate_model(self, data, model):
    model.eval()
    _, pred = model(data).max(dim=1)
    correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / data.test_mask.sum().item()
    print(f'Accuracy: {acc}\n')

    return acc

  def create_graph(self, ranked_list_path, features, labels):
    edge_index = self.compute_edge_index(ranked_list_path=ranked_list_path)
    edge_index = self.graph_augumentation(edge_index, features, labels)

    return edge_index

  def compute_edge_index(self, ranked_list_path) -> list:
    edge_index = []
    with open(ranked_list_path) as f:
      ranked_lists = []
      for line in f:
        ranked_lists.append([int(x) for x in line.strip().split()])

      for i, ranked_list in enumerate(ranked_lists):
        for node in ranked_list[:self.k + 1]:
          if i != node and i in ranked_lists[node][:self.k + 1]:
            edge_index.append((i, node))

    self.ranked_lists = ranked_lists
    return edge_index

  def graph_augumentation(self, edge_index, features, labels):
    x_test, y_test = self.get_test_features_and_labels(features, labels)

    cluster_labels, num_clusters = self.run_cluster(features, labels)
    clusters = self.get_clusters(cluster_labels, num_clusters)

    representative_nodes = self.get_representative_nodes(clusters, features)
    print(f'representative_nodes: {representative_nodes}\n')
    representative_node_acc_list = self.evaluate_representative_nodes(clusters, labels, representative_nodes)

    edge_index = self.create_sintetic_nodes(edge_index, features, labels, clusters, representative_nodes)

    return edge_index

  def get_test_features_and_labels(self, features, labels):
    x_test = [features[i] for i in range(len(features)) if self.test_mask[i]]
    y_test = [labels[i] for i in range(len(labels)) if self.test_mask[i]]

    return np.array(x_test), np.array(y_test)
  
  def run_cluster(self, features, labels):
    N_CLUSTERS = int(self.alpha * self.class_size)
    model = AgglomerativeClustering(
      n_clusters=N_CLUSTERS,
      metric=self.metric,
      linkage=self.linkage,
    )
    model = model.fit(features)

    # Evaluating clusters
    y_true = np.array(labels)
    nmi, vscore, acc = self.evaluate_clustering(y_true, model.labels_)
    cluster_measurements = [{'alpha': self.alpha, 'n_clusters': model.n_clusters_, 'nmi': nmi, 'vscore': vscore, 'accuracy': acc}]
    df = pd.DataFrame(cluster_measurements)
    print(f'{df}\n')

    print('Avaliando Cluster:')
    df_name = (
      'k-'+str(self.k)
      +'_metric-'+str(self.metric)
      +'_alpha-'+str(self.alpha)
      +'_link-'+str(self.linkage)
      +'_ClusterAval'
    )
    export_to_excel(df, df_name)

    print('Cluster Info:')
    print(f'n_clusters: {model.n_clusters_}')
    print(f'labels_: {model.labels_}')
    print(f'n_features_in: {model.n_features_in_}')
    print(f'n_connected_components: {model.n_connected_components_}\n')

    with np.printoptions(threshold=np.inf):
      print(f'cluster_labels = {model.labels_}, {model.labels_.shape}\n')

    return model.labels_, model.n_clusters_

  def evaluate_clustering(self, y_true, y_pred):
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred, average_method='min')
    vscore = metrics.v_measure_score(y_true, y_pred)
    acc = self.cluster_acc(y_true, y_pred)

    return nmi, vscore, acc
  
  def cluster_acc(self, y_true, y_pred):
    assert y_pred.size == y_true.size

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)

    for i in range(y_pred.size):
      w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    acc = np.sum([w[i, j] for i, j in zip(ind[0], ind[1])]) * 1.0 / y_pred.size

    return acc

  def get_clusters(self, cluster_labels, num_clusters):
    # Initializing clusters list
    clusters = [[] for _ in range(num_clusters)]

    # Grouping each node in its respective cluster
    for i, label in enumerate(cluster_labels):
      clusters[label].append(i)

    # Printing clusters
    print('------------------ Printing Clusters ------------------')
    for i, cluster in enumerate(clusters):
      print(f'Cluster {i}: {cluster}')
    print('\n')

    return clusters

  def get_representative_nodes(self, clusters, features):
    representative_nodes = []
    for cluster in clusters:
      sum_distances = [0] * len(cluster)
      for i in range(len(cluster)):
        for j in range(len(cluster)):
          sum_distances[i] += np.linalg.norm(features[cluster[i]] - features[cluster[j]])
      representative_node = cluster[np.argmin(sum_distances)]
      representative_nodes.append(representative_node)

    return representative_nodes

  def evaluate_representative_nodes(self, clusters, labels, representative_nodes):
    
    print('---------------- Avaliating Representative Nodes ----------------')

    dictionary = {
      'cluster':  [],
      'size':     [],
      'predicted':[],
      'acc':      [],
    }

    representative_nodes_acc_list = []
    for index, cluster in enumerate(clusters):
      representative_node_class = labels[representative_nodes[index]]
      print(f'y[representative_node]: {representative_node_class}')

      same_class_sum = 0
      for node in cluster:
        if labels[node] == representative_node_class:
          same_class_sum += 1

      dictionary['cluster'].append(index)
      dictionary['size'].append(len(cluster))
      dictionary['predicted'].append(same_class_sum)
      dictionary['acc'].append(round(float(same_class_sum)/float(len(cluster)), 4))

      acc = same_class_sum / len(cluster)
      print(f'same_class_sum: {same_class_sum}')
      print(f'cluster_size: {len(cluster)}')
      print(f'acc: {acc}\n')

      representative_nodes_acc_list.append(acc)
    
    df = pd.DataFrame(representative_nodes_acc_list, columns=['accuracy'])
    print('representative_nodes_acc_list')
    print(f'{df}\n')

    df_name = (
      'k-'+str(self.k)
      +'_met-'+str(self.metric)
      +'_alp-'+str(self.alpha)
      +'_link-'+str(self.linkage)
      +'_RepNodesAval.xlsx'
    )
    export_to_excel(df, df_name)
    dic_to_csv(dictionary, df_name)
    
    return representative_nodes_acc_list

  def create_sintetic_nodes(self, edge_index, features, labels, clusters, representative_nodes):
    print(f'--- ANTES DE CONECTAR ---\n')
    print(f'edge_index = {edge_index[:30]}')
    print(f'edge_index.shape = ({len(edge_index)},{len(edge_index[0])})\n')

    print(f'features = {features}')
    print(f'features.shape = ({len(features)},{len(features[-1])})\n')

    print(f'train_mask.shape = {len(self.train_mask)}')
    print(f'test_mask.shape = {len(self.test_mask)}\n')

    print(f'labels.shape = ({len(labels)})\n')

    new_features = []
    new_labels = []
    for index, cluster in enumerate(clusters):
      # representative node
      representative_node = representative_nodes[index]

      # syntetic node index and features
      synthetic_node_index = len(features - 1)
      synthetic_node_features = features[representative_node]

      # syntetic node label
      synthetic_node_label = labels[representative_node]
      if not self.train_mask[representative_node]:
        synthetic_node_label = self.infer_synthetic_node_label(cluster, labels, representative_node)

      # updating features, labels and maks
      new_features.append(synthetic_node_features)
      new_labels.append(synthetic_node_label)
      self.train_mask.append(True)
      self.test_mask.append(False)

      # updating edge_index -> connecting synthetic_node to all other nodes from its cluster
      for node in cluster:
        if node != representative_node:
          edge_index.append((synthetic_node_index, node))
      edge_index.append((representative_node, synthetic_node_index)) # linkando o representativo com o sintético --> ISSO É CORRETO??

    features = np.append(features, np.array(new_features), axis=0)
    labels = np.append(labels, np.array(new_labels), axis=0)
    self.features = features
    self.labels = labels

    print(f'--- DEPOIS DE CONECTAR ---\n')

    print(f'edge_index = {edge_index[:30]}')
    print(f'edge_index.shape = ({len(edge_index)},{len(edge_index[0])})\n')

    print(f'features = {features}')
    print(f'features.shape = ({len(features)},{len(features[-1])})\n')

    print(f'train_mask.shape = ({len(self.train_mask)})')
    print(f'test_mask.shape = ({len(self.test_mask)})\n')

    print(f'labels.shape = ({len(self.labels)})\n')

    return edge_index
  
  def infer_synthetic_node_label(self, cluster, labels, representative_node):
    class_votes = {}
    for node in cluster:
      if self.train_mask[node]:
        label = labels[node]
        if label in class_votes:
          class_votes[label] += 1
        else:
          class_votes[label] = 1

    if class_votes:
      return max(class_votes, key=class_votes.get)
    
    for rk_node in self.ranked_lists[representative_node]:
      if self.train_mask[rk_node]:
        return labels[rk_node]
      
    return labels[representative_node]