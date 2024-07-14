from main import GCN_Clustering
import numpy as np
import pandas as pd
from utils import export_to_excel

def load_masks(pN, classSize, trPerClass, valPerClass):
	train_mask = []
	val_mask = []
	test_mask = []

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
K=5
NETWORK = 'gcn'
ALPHA = 0.95
METRIC = 'euclidean'
LINKAGE = 'ward'

acc_list = []
for dataset in datasets:
	for index in range(len(dataset['features'])):
		print(f'feature: {dataset["features"][index]}')

		# Variables
		pN = dataset['pN'] # pn == n° de nós
		numberOfClasses = dataset['numberOfClasses']
		classSize = pN / numberOfClasses # clasSize == n° de nós rotulados para cada classe
		trPerClass = (classSize * 0.3) # quantos nós de TREINAMENTO para cada classe //0.18
		valPerClass = 0*trPerClass # quantos nós de VALIDAÇÃO para cada classe (* 0 se nao tiver validação)

		# Creating masks
		train_mask, val_mask, test_mask = load_masks(pN, classSize, trPerClass, valPerClass)
		print (f'train_mask: {train_mask[:20]}')
		print (f'test_mask: {test_mask[:20]}')
		print (f'val_mask: {val_mask[:20]}')
		print(f'train_mask.shape: {len(train_mask)}\n')

		# Loading classes
		y = load_classes(dataset['classes_file'], dataset['features'][index].split('/')[2])
		print(f'y: {y[:20]}')
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
			labels=y,
			ranked_list_path=dataset['ranked_lists'][index],
			num_classes=numberOfClasses,
		)

		acc_list.append({
			'dataset': dataset['features'][index].split('/')[2],
			'feature': dataset['features'][index].split('/')[4], 
			'acc': acc
		})

print(acc_list)
df = pd.DataFrame(acc_list)
print(df)
# export_to_excel(df, 'flowers17_features_acc')