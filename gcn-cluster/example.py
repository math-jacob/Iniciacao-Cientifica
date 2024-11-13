from main import GCN_Clustering
import numpy as np
import pandas as pd
from utils import export_to_excel

def load_masks(pN, classSize, tr_prop, val_prop):
	trPerClass = int(classSize * tr_prop)
	valPerClass = int(classSize * val_prop)
	train_mask, val_mask, test_mask = [], [], []

	for i in range (pN):
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

	return train_mask, val_mask, test_mask

def load_classes(y_file, dataset_name):
	y = []
	with open(y_file) as f:
		for line in f:
			line_list = line.strip().split(':')
			if dataset_name == 'cub200':
				y.append(int(line_list[1]) - 1)
			else:
				y.append(int(line_list[1]))
	
	return y

def load_feature_matrix(feat_matrix_file):
	return np.load(feat_matrix_file)

def get_hyperparameters(dataset_name):
	# Example function returning hyperparameters based on the dataset

	# --------- Hiperparametros do CVIU ----------
	# tr_prop = 0.1, tr_test = 0.9
	# 10e-4 learning
	# 256 Neuronios
	# k=40 reciproco
	# ----------------------------------------------

	hyperparameters = {
		'flowers17': {'tr_prop': 0.1, 'val_prop': 0, 'pNNeurons': 256, 'pNEpochs': 200, 'pLR': 0.0001},
		'dogs': {'tr_prop': 0.1, 'val_prop': 0, 'pNNeurons': 256, 'pNEpochs': 200, 'pLR': 0.0001},
		'corel5k': {'tr_prop': 0.4, 'val_prop': 0, 'pNNeurons': 256, 'pNEpochs': 200, 'pLR': 0.001},
		'cub200': {'tr_prop': 0.4, 'val_prop': 0, 'pNNeurons': 256, 'pNEpochs': 300, 'pLR': 0.0001},
	}
	return hyperparameters.get(dataset_name, {'tr_prop': 0.2, 'val_prop': 0.1, 'pNNeurons': 32, 'pNEpochs': 200, 'pLR': 0.001})

datasets = [
	{
		'pN': 1360,
		'numberOfClasses': 17,
		'classes_file': './datasets/oxford17flowers/flowers_classes.txt',
		'features': [
			'./datasets/oxford17flowers/features/resnet152/features.npy',
			'./datasets/oxford17flowers/features/dpn92/features.npy',
			'./datasets/oxford17flowers/features/senet154/features.npy',
			'./datasets/oxford17flowers/features/vitb16/features.npy',
		],
		'ranked_lists': [
			'./datasets/oxford17flowers/ranked-lists/resnet.txt',
			'./datasets/oxford17flowers/ranked-lists/dpn.txt',
			'./datasets/oxford17flowers/ranked-lists/senet.txt',
			'./datasets/oxford17flowers/ranked-lists/vitb.txt',
		]
	},
	# {
	# 	'pN': 20580,
	# 	'numberOfClasses': 120,
	# 	'classes_file': './datasets/dogs/dogs_classes.txt',
	# 	'features': [
	# 		'./datasets/dogs/features/resnet152/features.npy',
	# 		'./datasets/dogs/features/dpn92/features.npy',
	# 		'./datasets/dogs/features/senet154/features.npy',
	# 	],
	# 	'ranked_lists': [
	# 		'./datasets/dogs/ranked-lists/resnet.txt',
	# 		'./datasets/dogs/ranked-lists/dpn.txt',
	# 		'./datasets/dogs/ranked-lists/senet.txt',
	# 	]
	# },
	# {
	# 	'pN': 5000,
	# 	'numberOfClasses': 50,
	# 	'classes_file': './datasets/corel5k/corel5k_classes.txt',
	# 	'features': [
	# 		'./datasets/corel5k/features/resnet152/features.npy',
	# 		'./datasets/corel5k/features/dpn92/features.npy',
	# 		'./datasets/corel5k/features/senet154/features.npy',
	# 		'./datasets/corel5k/features/vitb16/features.npy',
	# 	],
	# 	'ranked_lists': [
	# 		'./datasets/corel5k/ranked-lists/resnet.txt',
	# 		'./datasets/corel5k/ranked-lists/dpn.txt',
	# 		'./datasets/corel5k/ranked-lists/senet.txt',
	# 		'./datasets/corel5k/ranked-lists/vitb.txt',
	# 	]
	# },
	# {
	# 	'pN': 11788,
	# 	'numberOfClasses': 200,
	# 	'classes_file': './datasets/cub200/cub200_classes.txt',
	# 	'features': [
	# 		'./datasets/cub200/features/resnet152/features.npy',
	# 		'./datasets/cub200/features/dpn92/features.npy',
	# 		'./datasets/cub200/features/senet154/features.npy',
	# 		'./datasets/cub200/features/vitb16/features.npy',
	# 	],
	# 	'ranked_lists': [
	# 		'./datasets/cub200/ranked-lists/resnet.txt',
	# 		'./datasets/cub200/ranked-lists/dpn.txt',
	# 		'./datasets/cub200/ranked-lists/senet.txt',
	# 		'./datasets/cub200/ranked-lists/vitb.txt',
	# 	]
	# },
]

# Parameters
K=40
NETWORK = 'gcn'
ALPHA = 0.2
METRIC = 'euclidean'
LINKAGE = 'ward'

for dataset in datasets:
	acc_list = []
	for index in range(len(dataset['features'])):
		for alpha in [0.2, 0.4, 0.8]:
			ALPHA = alpha

			print(f'feature: {dataset["features"][index]}')

			# Variables
			dataset_name = dataset['features'][index].split('/')[2]
			pN = dataset['pN']
			numberOfClasses = dataset['numberOfClasses']
			classSize = pN // numberOfClasses
			hyperparameters = get_hyperparameters(dataset_name)

			# Creating masks
			train_mask, val_mask, test_mask = load_masks(pN, classSize, hyperparameters['tr_prop'], hyperparameters['val_prop'])

			print (f'train_mask: {train_mask[:20]}')
			train_count = sum(1 for node in train_mask if node)
			print (f'train_mask.shape: ({train_count})\n')

			print (f'test_mask: {test_mask[:20]}')
			test_count = sum(1 for node in test_mask if node)
			print (f'test_mask.shape: ({test_count})\n')

			print (f'val_mask: {val_mask[:20]}')
			val_count = sum(1 for node in val_mask if node)
			print(f'val_mask.shape: {val_count}\n')

			# Loading classes
			y = load_classes(dataset['classes_file'], dataset_name)
			print(f'len(y): {len(y)}\n')

			# Loading feature matrix
			x = load_feature_matrix(dataset['features'][index])
			print(f'feat_matrix: {x}\n')

			# ------------------------- Run GCN Clustering Method ------------------------
			gcn_clustering = GCN_Clustering(
				train_mask=train_mask,
				test_mask=test_mask,
				val_mask=[],
				class_size=classSize,
				k=K,
				metric=METRIC,
				network=NETWORK,
				alpha=ALPHA,
				linkage=LINKAGE
			)

			acc = gcn_clustering.run(
				features=x,
				feature_name=dataset['features'][index].split('/')[4],
				labels=y,
				ranked_list_path=dataset['ranked_lists'][index],
				num_classes=numberOfClasses,
				apply_cluster=True,
			)

			acc_list.append({
				'dataset': dataset_name,
				'feature': dataset['features'][index].split('/')[4],
				'alpha': ALPHA,
				'accuracy': acc
			})

	df = pd.DataFrame(acc_list)
	print(df)

	df_name = (
		'feat-'+dataset['features'][index].split('/')[4]
		+'_k-'+str(K)
		+'_met-'+str(METRIC)
		+'_alp-'+str(ALPHA)
		+'_link-'+str(LINKAGE)
		+'_GCNClusteringAcc'
	)
	export_to_excel(df, df_name)