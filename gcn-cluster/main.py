# Utility packages
import time
import numpy as np
import pandas as pd
from utils import export_to_excel, dic_to_csv

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
  
  def run(
    self, 
    features: np.array, 
    labels: np.array, 
    ranked_list_path: str,
    num_classes: int, 
    ):
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Edge index
    edge_index = self.create_graph(ranked_list_path=ranked_list_path, features=features, labels=labels)
    edge_index = torch.tensor(edge_index)
    edge_index = edge_index.t().contiguous().to(device)

    # Data definition
    x = torch.tensor(self.features).to(device)
    y = torch.tensor(self.labels, dtype=torch.long).to(device)

    # Mask definition
    train_mask = torch.tensor(self.train_mask).to(device)
    test_mask = torch.tensor(self.test_mask).to(device)
    val_mask = torch.tensor(self.val_mask).to(device)

    # -------------------------------------------------- PARTE QUE IMPLEMENTA A GCN --------------------------------------------------

    # Tensor data
    data = Data(
      x=x.float(),
      edge_index=edge_index, 
      y=y,
      train_mask=train_mask,
      test_mask=test_mask
    ) # está sem validação

    # Variables
    pNNeurons = 32
    pNEpochs = 200
    pNFeatures = len(features[0])
    pLR = 0.001
    NUM_EXECS = 10
    
    # Defining GCN Model
    class Net(torch.nn.Module):
      def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(pNFeatures, pNNeurons)
        self.conv2 = GCNConv(pNNeurons, num_classes)

      def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
    
    # Model and optimizer
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=pLR, weight_decay=5e-4)

    # Training
    acc_list = []
    for exec in range(NUM_EXECS):
      model.train()
      for epoch in range(pNEpochs):	
        optimizer.zero_grad()
        out = model(data)

        # Overfit checking
        # _, pred = out.max(dim=1)
        # correct = float(pred[data.train_mask]
        #   .eq(data.y[data.train_mask])
        #   .sum()
        #   .item())
        # acc = correct / data.train_mask.sum().item()
        # if acc == 1.0:
        #   print(f"Early stoping on epoch {epoch}")
        #   break

        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
      
      model.eval()
      _, pred = model(data).max(dim=1)

      # Training accuracy
      correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
      acc = correct / data.test_mask.sum().item()
      acc_list.append(acc)
      # print(f'acc_list = {acc_list}\n')

    print(f'Accuracy: {sum(acc_list)/NUM_EXECS}')

  def create_graph(self, ranked_list_path, features, labels):
    edge_index = self.compute_edge_index(ranked_list_path=ranked_list_path)
    edge_index = self.graph_augumentation(edge_index, features, labels)

    return edge_index

  def compute_edge_index(self, ranked_list_path) -> list:
    edge_index = []

    with open(ranked_list_path) as f:
      line_list_matrix = []
      ranked_lists = []

      for line in f:
        line_list = line.strip().split(' ')
        line_list_matrix.append(line_list)
        ranked_lists.append([int(x) for x in line_list])
		
      for i in range(0, len(line_list_matrix)):
        l = self.k
        j = 0
        while j < l and j < len(line_list_matrix[i]):
          # print(f'\nTeste {(i,line_list_matrix[i][j])}')
          if (i != int(line_list_matrix[i][j])) and (str(i) in line_list_matrix[int(line_list_matrix[i][j])][0:self.k+1]):
            # print('PASSOU')
            # print(f'{int(line_list_matrix[i][j])} = {line_list_matrix[int(line_list_matrix[i][j])][0:k+1]}\n')
            # time.sleep(1)
            
            edge_index.append((i, int(line_list_matrix[i][j])))
          elif i == int(line_list_matrix[i][j]):
            l += 1
          # else:
            # print('NAO PASSOU')
            # print(f'{int(line_list_matrix[i][j])} = {line_list_matrix[int(line_list_matrix[i][j])][0:k+1]}\n')
          j += 1

    self.ranked_lists = ranked_lists

    return edge_index

  def graph_augumentation(self, edge_index, features, labels):
    x_test, y_test = self.get_test_features_and_labels(features, labels)

    cluster_labels, num_clusters = self.run_cluster(features, labels)

    clusters = self.get_clusters(cluster_labels, num_clusters)

    representative_nodes = self.get_representative_nodes(clusters, features)
    print(f'representative_nodes: {representative_nodes}\n')

    print('Verificação das classes de um cluster específico')
    for index in clusters[1]:
      print(labels[index])

    # Testando meu algoritmo -> avaliação do nó representativo
    representative_node_acc_list = self.avaliate_representative_nodes(clusters, labels, representative_nodes)

    edge_index = self.create_sintetic_nodes(edge_index, features, labels, clusters, representative_nodes)

    return edge_index

  def get_test_features_and_labels(self, features, labels):
    x_test = []
    y_test = []
    x_aux_list = []
    for i in range(len(features)):
      if self.test_mask[i]:
        for j in range(len(features[i])):
          x_aux_list.append(features[i][j])
        
        y_test.append(labels[i])
        x_test.append(x_aux_list.copy())
        x_aux_list.clear()

    x_test = np.array(x_test)
    print(f'x_test: {x_test}')
    print(f'x_test.shape: ({len(x_test)},{len(x_test[0])})\n')

    y_test = np.array(y_test)
    print(f'y_test: {y_test}')
    print(f'y_test.shape: ({len(y_test)})\n')

    return x_test, y_test
  
  def run_cluster(self, features, labels):

    # Validation functions
    def evaluate(y_true, y_pred):
      nmi = metrics.normalized_mutual_info_score(
        y_true,
        y_pred,
        average_method='min'
      )
      
      vscore = metrics.v_measure_score(y_true, y_pred)
      acc = cluster_acc(y_true, y_pred)

      return nmi, vscore, acc
  
    def cluster_acc(y_true, y_pred):
      """
      Calculate clustering accuracy. Require scikit-learn installed
      # Arguments
          y: true labels, numpy.array with shape `(n_samples,)`
          y_pred: predicted labels, numpy.array with shape `(n_samples,)`
      # Return
          accuracy, in [0,1]
      """

      # print(y_pred.size, y_true.size()[0])
      assert y_pred.size == y_true.size

      D = max(y_pred.max(), y_true.max()) + 1
      w = np.zeros((D, D), dtype=np.int64)
      for i in range(y_pred.size):
          w[y_pred[i], y_true[i]] += 1
      ind = linear_assignment(w.max() - w)
      acc = np.sum([w[i, j] for i, j in zip(ind[0], ind[1])]) * 1.0 / y_pred.size
      return acc

    # Parameters
    #N_CLUSTERS = 16
    #METRIC = 'euclidean'
    #LINKAGE = 'ward'

    #alpha = 0.95
    cluster_measurements = []
    aux_measures = {}
    index = 0

    # while alpha < 1:
    #N_CLUSTERS = int(alpha * self.class_size)
    N_CLUSTERS = int(self.alpha * self.class_size)

    model = AgglomerativeClustering(
      n_clusters=N_CLUSTERS,
      metric=self.metric,
      linkage=self.linkage,
    )
    model = model.fit(features)

    # plt.title("Hierarchical Clustering Dendrogram")
    # # plot the top three levels of the dendrogram
    # plot_dendrogram(model, truncate_mode="level", p=3)
    # plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    # plt.show()

    # Avaliating clusters
    y_true = np.array(labels)
    # print(f'y_true: {y_true}')
    # print(f'y_pred: {model.labels_}')
    # print(f'len(y_true): {len(y_true)}')
    # print(f'len(y_pred): {len(model.labels_)}\n')

    nmi, vscore, acc = evaluate(y_true, model.labels_)
    # print("NMI ->", nmi)
    # print("V_Measure ->", vscore)
    # print("Accuracy ->", acc)

    aux_measures['alpha'] = self.alpha
    aux_measures['n_clusters'] = N_CLUSTERS
    aux_measures['nmi'] = nmi
    aux_measures['vscore'] = vscore
    aux_measures['accuracy'] = acc

    cluster_measurements.append(aux_measures.copy())

    #alpha += 0.05
    #index += 1
    aux_measures.clear()

    df = pd.DataFrame(cluster_measurements)
    print('Avaliando Cluster:')
    print(f'{df}\n')
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

    for i_cluster in range(0,len(clusters)):
      sum_distances = [0] * len(clusters[i_cluster])

      for i in range(len(clusters[i_cluster])):
        for j in range(len(clusters[i_cluster])):
          sum_distances[i] += math.sqrt(
            sum(
              (features[clusters[i_cluster][i]][k] - features[clusters[i_cluster][j]][k])**2
              for k in range(len(features[clusters[i_cluster][i]]))
            )
          )

      representative_node = clusters[i_cluster][sum_distances.index(min(sum_distances))]
      representative_nodes.append(representative_node)

    return representative_nodes

  def avaliate_representative_nodes(self, clusters, labels, representative_nodes):
    
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
        # print(f'node: {node}, y[node]: {labels[node]}')
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

    new_features = []
    new_labels = []
    for index, cluster in enumerate(clusters):
      # representative node
      representative_node = representative_nodes[index]

      # sintetic node index and features
      sintetic_node_index = len(features - 1)
      sintetic_node_features = features[representative_node]

      # sintetic node label
      if self.train_mask[representative_node]:
        sintetic_node_class = labels[representative_node]
      elif self.test_mask[representative_node]:
        # print(f'{representative_node} TESTE')

        classes_found = {}
        for node in cluster:
          if self.train_mask[node]:
            if labels[node] in classes_found.keys():
              classes_found[labels[node]] += 1
            else:
              classes_found[labels[node]] = 1
        
        # print(cluster)
        # print(classes_found)

        if (classes_found):
          most_voted_class = max(classes_found, key=classes_found.get) 
          sintetic_node_class = most_voted_class
          # print(f'Most voted class: {most_voted_class}\n')
        else:
          # look into ranked list
          for rk_node in self.ranked_lists[representative_node]:
            if self.train_mask[rk_node]:
              sintetic_node_class = labels[rk_node]
              break

      # updating features, labels and maks
      new_features.append(sintetic_node_features)
      new_labels.append(sintetic_node_class)
      self.train_mask.append(True)
      self.test_mask.append(False)

      # updating edge_index -> connecting sintetic_node to all other nodes from its cluster
      for node in cluster:
        if node != representative_node:
          edge_index.append((sintetic_node_index, node))

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

    print(f'labels.shape = ({len(labels)})\n')

    return edge_index