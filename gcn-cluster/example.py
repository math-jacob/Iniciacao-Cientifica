from main import GCN_Clustering
import numpy as np

# Parameters
K=5
NETWORK = 'gcn'
METRIC = 'euclidean'
numberOfClasses = 17

# ------------------------- Load Masks ------------------------

# Variables
pN = 1360 # pn == n° de nós
classSize = 80 # clasSize == n° de nós rotulados para cada classe
trPerClass = 15 # quantos nós de TREINAMENTO para cada classe
valPerClass = 0*trPerClass # quantos nós de VALIDAÇÃO para cada classe (* 0 se nao tiver validação)

train_mask = []
val_mask = []
test_mask = []

for i in range (pN):
	# o símbolo // representa floor division
	d = i % classSize
	if (d<trPerClass):
		valueTr = True
		valueVal = False
	else:
		valueTr = False
		if (d<(valPerClass+trPerClass)):
			valueVal = True
		else:
			valueVal = False
	valueTest = (not valueTr) and (not valueVal)
	
	train_mask.append(valueTr)
	test_mask.append(valueTest)
	val_mask.append(valueVal)

print (f'train_mask: {train_mask[:20]}')
print (f'test_mask: {test_mask[:20]}')
print (f'val_mask: {val_mask[:20]}\n')

# ------------------------- Load Classes ------------------------

y_file = './oxford17flowers-driveIC/flowers_classes.txt'
y = []
with open(y_file) as f:
	for line in f:
		line_list = line.strip().split(':')
		y.append(int(line_list[1]))
		
print(f'y: {y[:20]}')
print(f'len(y): {len(y)}\n')

#-------------------------------- Load Feature Matrix -------------------------------------
feat_matrix_file = './oxford17flowers-driveIC/cnn-last-linear-resnet/features.npy'
x = np.load(feat_matrix_file)

print(f'feat_matrix: {x}\n')

# ------------------------- Run GCN Clustering Method ------------------------

gcn_clustering = GCN_Clustering(
	train_mask=train_mask,
	test_mask=test_mask,
	val_mask=[],
	class_size=classSize,
	k=K, 
  network=NETWORK,
	metric=METRIC
)

gcn_clustering.run(
	features=x,
	labels=y,
	ranked_list_path='./oxford17flowers-driveIC/ranked-lists-CNN-ResNet.txt',
	num_classes=numberOfClasses,
)