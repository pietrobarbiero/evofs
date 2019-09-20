# This script has been designed to perform multi-objective learning of feature selection 
# by Alberto Tonda and Pietro Barbiero, 2018 <alberto.tonda@gmail.com> <pietro.barbiero@studenti.polito.it>

#basic libraries
import argparse
import copy
import datetime
import inspyred
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import random
import os
import sys
import logging
import time
import multiprocessing
import csv
import continuous
from decimal import Decimal


# sklearn library
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif, f_classif
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.datasets.samples_generator import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

import openml

from itertools import combinations
from pandas import read_csv
from scipy.stats.kde import gaussian_kde

import seaborn as sns
from matplotlib.colors import ListedColormap

#matplotlib.rcParams.update({'font.size': 15})

import warnings
warnings.filterwarnings("ignore")

#matplotlib.rcParams['text.usetex'] = True
#matplotlib.rcParams['text.latex.unicode'] = True

def main(selectedDataset, pop_size, max_generations, did, method, folder_name):
	
	# a few hard-coded values
	figsize = [5, 4]
	seed = 42
	offspring_size = 2 * pop_size
	n_splits = 10
	
	standard_methods = [ 'mi', 'anova', 'RFE', ]

	# a list of classifiers
	allClassifiers = [
#			[RandomForestClassifier, "RandomForestClassifier", 1],
			[LogisticRegression, "LogisticRegression", 1],
#			[BaggingClassifier, "BaggingClassifier", 1],
#			[SVC, "SVC", 1],
#			[RidgeClassifier, "RidgeClassifier", 1],
			]
	
	selectedClassifiers = [classifier[1] for classifier in allClassifiers]
	
	# load different datasets, prepare them for use
	# load synthetic datasets
#	iris = datasets.load_iris()
	# load more serious datasets
	db = openml.datasets.get_dataset(did)
	X, y, attribute_names = db.get_data(
			   target=db.default_target_attribute,
			      return_attribute_names=True)
	dataList = [
#				[iris.data, iris.target, 0, "iris4"],
#		        [iris.data[:, 2:4], iris.target, 0, "iris2"],
				[X, y, 0, selectedDataset]
		      ]

	# argparse; all arguments are optional
	parser = argparse.ArgumentParser()
	
	parser.add_argument("--classifiers", "-c", nargs='+', help="Classifier(s) to be tested. Default: %s. Accepted values: %s" % (selectedClassifiers[0], [x[1] for x in allClassifiers]))
	parser.add_argument("--dataset", "-d", help="Dataset to be tested. Default: %s. Accepted values: %s" % (selectedDataset,[x[3] for x in dataList]))
	
	parser.add_argument("--pop_size", "-p", type=int, help="EA population size. Default: %d" % pop_size)
	parser.add_argument("--offspring_size", "-o", type=int, help="Ea offspring size. Default: %d" % offspring_size)
	parser.add_argument("--max_generations", "-mg", type=int, help="Maximum number of generations. Default: %d" % max_generations)
	
	# finally, parse the arguments
	args = parser.parse_args()
	
	# a few checks on the (optional) inputs
	if args.dataset : 
		selectedDataset = args.dataset
		if selectedDataset not in [x[3] for x in dataList] :
			print("Error: dataset \"%s\" is not an accepted value. Accepted values: %s" % (selectedDataset, [x[3] for x in dataList]))
			sys.exit(0)
	
	if args.classifiers != None and len(args.classifiers) > 0 :
		selectedClassifiers = args.classifiers
		for c in selectedClassifiers :
			if c not in [x[1] for x in allClassifiers] :
				print("Error: classifier \"%s\" is not an accepted value. Accepted values: %s" % (c, [x[1] for x in allClassifiers]))
				sys.exit(0)
	
	if args.max_generations : max_generations = args.max_generations
	if args.pop_size : pop_size = args.pop_size
	if args.offspring_size : offspring_size = args.offspring_size
	
	# TODO: check that min_points < max_points and max_generations > 0
	
	
	# print out the current settings
	print("Selected dataset: %s" % (selectedDataset))

	# create the list of classifiers
	classifierList = [ x for x in allClassifiers if x[1] in selectedClassifiers ]	

	# pick the dataset 
	db_index = -1
	for i in range(0, len(dataList)) :
		if dataList[i][3] == selectedDataset :
			db_index = i
	dbname = dataList[db_index][3]
	X, y = dataList[db_index][0], dataList[db_index][1]
	number_classes = np.unique(y).shape[0]
	print("Number of samples: %d; number of features: %d; number of classes: %d; number of splits: %d"
			   % (X.shape[0], X.shape[1], number_classes, n_splits))
		
	for classifier in classifierList:
		
		classifier_class = classifier[0]
		classifier_name = classifier[1]
		classifier_type = classifier[2]
		
		# initialize classifier; some classifiers have random elements, and
		# for our purpose, we are working with a specific instance, so we fix
		# the classifier's behavior with a random seed
		if classifier_type == 1: model = classifier_class(random_state=seed)
		else : model = classifier_class()
		
		print("Classifier used: " + classifier_name)

#		# start creating folder name
#		experiment_name = os.path.join(folder_name, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) + "-feature-selection-evolution-" + dbname + "-" + classifier_name
#		if not os.path.exists(experiment_name) : os.makedirs(experiment_name)
		
		# print csv header
		with open( os.path.join(folder_name, 
						  "final_results_" + dbname + "_" + method + "_" + classifier_name + ".csv"), "w") as fp :
			fp.write("feature-set,size,test-error,fold\n")
		
		# create splits for cross-validation
		print("Creating train/test split...")
		skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
		
		split_index = 1
		for train_index, test_index in skf.split(X, y) :
			
#			if split_index <= 6:
#				split_index = split_index + 1
#				continue
		
			X_train, y_train = X[train_index], y[train_index]
			X_test, y_test = X[test_index], y[test_index]
			print("\tSplit %d" %(split_index))
		
			# rescale data
			scaler = StandardScaler()
			sc = scaler.fit(X_train)
#			X_t = sc.transform(X)
			X_train = sc.transform(X_train)
			X_test = sc.transform(X_test)
			
			print("Feature selection algorithm: %s" % (method))
			
			# standard methods
			if method in standard_methods:
				
				if method == 'mi': fs_class = mutual_info_classif
				elif method == 'anova': fs_class = f_classif
				elif method == 'RFE': fs_class = RFE
				
				feature_set, test_error = standard_feature_selection(X_train, y_train, X_test, y_test, 
													 model, fs_class, dbname, classifier_name, method, split_index)
				
			elif method == 'EvoFS':
				
				# seed the initial population using MI
				initial_pop = []
#				for k in range(1, X_train.shape[1]+1):
#					# MI individuals
#					fs_mi = SelectKBest(mutual_info_classif, k=k)
#					fs_mi.fit(X_train, y_train)
#					# RFE individuals
#					referenceClassifier = copy.deepcopy(model)
#					fs_rfe = RFE(referenceClassifier, k, step=1)
#					fs_rfe.fit(X_train, y_train)
#					# ANOVA individuals
#					fs_anova = SelectKBest(f_classif, k=k)
#					fs_anova.fit(X_train, y_train)
#					
#					initial_pop.append( list( fs_mi.get_support() ) )
#					initial_pop.append( list( fs_rfe.get_support() ) )
#					initial_pop.append( list( fs_anova.get_support() ) )
				
				# start evolving feature selection
				start = time.time()
				final_archive, trainAccuracy = evolveFeatureSelection(X_train, y_train, 
												   model, pop_size, offspring_size, max_generations, 
												   initial_pop=initial_pop,
												   seed=seed, experiment_name=folder_name)
				end = time.time()
				exec_time = end - start
				
#				final_archive = sorted(final_archive, key = lambda x : x.fitness[0])
				feature_set = []
				test_error = []
				for solution in final_archive:
					
					# retain only the features selected according to the candidate solution
					cAsBoolArray = np.array(solution.candidate, dtype=bool)
					X_train_reduced = X_train[:, cAsBoolArray]
					X_test_reduced = X_test[:, cAsBoolArray]
					# compute classification accuracy
					referenceClassifier = copy.deepcopy(model)
					referenceClassifier.fit(X_train_reduced, y_train)
					accuracy = referenceClassifier.score( X_test_reduced, y_test )
					test_error.append( 1-accuracy )
					
					# save feature set
					sol = []
					j = 0
					for i in solution.candidate:
						if i == 1:
							sol.append(j)
						j = j + 1
					feature_set.append( tuple(sol) )
			
			with open( os.path.join(folder_name, 
							  "final_results_" + dbname + "_" + method + "_" + classifier_name + ".csv"), "a") as fp :
				for fs, err in zip(feature_set, test_error) :
					fp.write(str(fs).replace(","," ")) # replace ',' with ' '  in string representation of a list, avoid issues with CSV
					fp.write("," + str(len(fs)))
					fp.write("," + str(err))
					fp.write("," + str(split_index))
					fp.write("\n")
			
			split_index = split_index + 1

	return

# function that does most of the work
def evolveFeatureSelection(X, y,
						   classifier, pop_size, offspring_size, max_generations, 
						   initial_pop, seed=None, experiment_name=None) :
	
	# initialize pseudo-random number generation
	prng = random.Random()
	prng.seed(seed)

	# set the reference performance as the one using all features
	print("Computing initial classifier performance using all features...")
	referenceClassifier = copy.deepcopy(classifier)
	referenceClassifier.fit(X, y)
	y_train_pred = referenceClassifier.predict(X)
	trainAccuracy = accuracy_score(y, y_train_pred)
	print("Initial performance: train=%.4f" % (trainAccuracy))

	# set up MOEA
	print("\nSetting up evolutionary algorithm...")
	ea = inspyred.ec.emo.NSGA2(prng)
	ea.variator = [ variateFeatureSelection ]
	ea.terminator = inspyred.ec.terminators.generation_termination
	ea.observer = observeFeatureSelection
	maximize = False
	n_classes = len( np.unique(y) )
	
	# start evolution
	final_population = ea.evolve(    
					generator = generateFeatureSelection,
					
					evaluator = evaluateFeatureSelection,
					
					# this part is defined to use multi-process evaluations
#					evaluator = inspyred.ec.evaluators.parallel_evaluation_mp,
#					mp_evaluator = evaluateFeatureSelection, 
#					mp_num_cpus = multiprocessing.cpu_count()-2,
					
					pop_size = np.max([ len(initial_pop), pop_size ]),
					num_selected = offspring_size,
					maximize = maximize, 
					max_generations = max_generations,
					
					# extra arguments here
					n_classes = n_classes,
					classifier = classifier,
					X_train = X,
					y_train = y,
					experimentName = experiment_name,
					current_time = datetime.datetime.now(),
					numG = 0,
					seed = initial_pop)

	return ea.archive, trainAccuracy

# initial random generation of core sets (as binary strings)
def generateFeatureSelection(random, args) :

	# set up some parameters
	X_train = args.get("X_train", None)
	min_features = 1
	max_features = X_train.shape[1]
	individual_length = X_train.shape[1]
	
	# create an empty individual
	individual = [0] * individual_length
	
	# fill the individual randomly
	n_features = random.randint( min_features, max_features )
	for i in range(n_features) :
		random_index = random.randint(0, individual_length-1)
		individual[random_index] = 1
	
	return individual

@inspyred.ec.variators.crossover
def variateFeatureSelection(random, parent1, parent2, args) :
	
	X_train = args.get("X_train", None)
	min_features = 1
	max_features = X_train.shape[1]
	
	# well, for starters we just crossover two individuals, then mutate
	children = [ list(parent1), list(parent2) ]
	
	# one-point crossover!
	cutPoint = random.randint(0, len(children[0])-1)
	for index in range(0, cutPoint+1) :
		temp = children[0][index]
		children[0][index] = children[1][index]
		children[1][index] = temp 
	
	# mutate!
	for child in children : 
		mutationPoint = random.randint(0, len(child)-1)
		if child[mutationPoint] == 0 :
			child[mutationPoint] = 1
		else :
			child[mutationPoint] = 0
	
	# check if individual is still valid, and (in case it isn't) repair it
	for child in children :
		
		n_features = [ index for index, value in enumerate(child) if value == 1 ]
		
		while len(n_features) > max_features :
			index = random.choice( n_features )
			child[index] = 0
			n_features = [ index for index, value in enumerate(child) if value == 1 ]
		
		if len(n_features) < min_features :
			index = random.choice( [ index for index, value in enumerate(child) if value == 0 ] )
			child[index] = 1
			n_features = [ index for index, value in enumerate(child) if value == 1 ]
	
	return children

# function that evaluates the core sets
#@inspyred.ec.utilities.memoize
def evaluateFeatureSelection(candidates, args) :
	
	# retrieve useful variables
	X = args.get("X_train", None)
	y = args.get("y_train", None)
	classifier = args.get("classifier", None)
	num_generations = args.get("numG", None)
	
	seed = 42
	index = 1
	fitness = []
	
	# compute the fitness of each individual
	for c in candidates :
		
		cAsBoolArray = np.array(c, dtype=bool)
		
		# compute the number of selected features
		nSelectedFeatures = np.sum( cAsBoolArray )
			
		# compute individual feature importance
		F, pval = f_classif(X[:, cAsBoolArray], y)
		pval[ np.isnan(pval)==True ] = 0.05
#		statTest = np.sum(pval)
		statTest = np.max(pval)
		
		# split the data into training and validation sets
		skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=num_generations*index)
		accuracy = []
		for train_index, val_index in skf.split(X, y) :
			
			X_train, y_train = X[train_index], y[train_index]
			X_val, y_val = X[val_index], y[val_index]
			
			# retain only the features selected according to the candidate solution
			X_train_reduced = X_train[:, cAsBoolArray]
			X_val_reduced = X_val[:, cAsBoolArray]
			
			# compute classification accuracy
			referenceClassifier = copy.deepcopy(classifier)
			referenceClassifier.fit(X_train_reduced, y_train)
			accuracy.append( referenceClassifier.score( X_val_reduced, y_val ) )
		
		accuracy = np.mean( accuracy )
		error = 1-accuracy
		
#		fitness.append( inspyred.ec.emo.Pareto( [nSelectedFeatures, error, statTest] ) )
		fitness.append( inspyred.ec.emo.Pareto( [statTest, error] ) )
		
		index = index + 1
	
	return fitness

# the 'observer' function is called by inspyred algorithms at the end of every generation
def observeFeatureSelection(population, num_generations, num_evaluations, args) :
	
	old_time = args["current_time"]
	current_time = datetime.datetime.now()
	delta_time = current_time - old_time 
	
	# I don't like the 'timedelta' string format, so here is some fancy formatting
	delta_time_string = str(delta_time)[:-7] + "s"
	
	pop = sorted(population, key = lambda x : x.fitness[1])
	
#	print("[%s] Generation %d, Best individual: selected features=%.2f, error=%.4f, pval=%.2E" 
#				   % (delta_time_string, num_generations, pop[0].fitness[0], pop[0].fitness[1], Decimal(str(pop[0].fitness[2])) ))
	
	print("[%s] Generation %d, Best individual: pval=%.4E, error=%.4f" 
				   % (delta_time_string, num_generations, Decimal(str(pop[0].fitness[0])), pop[0].fitness[1]))
	
	args["current_time"] = current_time
	args["numG"] = num_generations

	return

def standard_feature_selection(X_train, y_train, X_test, y_test, model, fs_class, dbname, classifier_name, method, split_index):
	
	feature_set = []
	errors = []
	
	start = time.time()
	for k in range(1, X_train.shape[1]+1):
		
		if method == 'RFE':
			referenceClassifier = copy.deepcopy(model)
			fs = RFE(referenceClassifier, k, step=1)
		else:
			fs = SelectKBest(fs_class, k=k)
		X_train_reduced = fs.fit_transform(X_train, y_train)
		X_test_reduced = fs.fit_transform(X_test, y_test)
		feature_set.append( tuple(fs.get_support(indices=True)) )
		
		referenceClassifier = copy.deepcopy(model)
		referenceClassifier.fit(X_train_reduced, y_train)
		errors.append( 1-referenceClassifier.score( X_test_reduced, y_test ) )
		
		print("\t[%s] Error using %d features: %.4f" % (method, k, 1-referenceClassifier.score( X_test_reduced, y_test )))
		
	end = time.time()
	exec_time = end - start
	
	return feature_set, errors
		
def cfs_fitness(k, r_ff, r_cf):
	# compute the correlation-based feature selection fitness
	return ( k * r_cf ) / np.sqrt( k + k*(k-1) * r_ff )
	
def r_cf(positions, h, hy, joint_hy, conditional_hy):
	# compute the average feature-class correlation
	r_cf = []
	for i in positions:
		num = h[i] + hy - joint_hy[i]
		den = h[i] + hy
		r_cf.append( 2 * num / den )
	r_cf = np.mean(r_cf)
	return r_cf
	
def r_ff(positions, h, joint_h, conditional_h):
	# compute all possible pairs of features
	pair_list = list( combinations(positions, 2) )
	
	# compute the average feature-feature correlation
	r_ff = []
	for i, j in pair_list:
		num = h[i] + h[j] - joint_h[i, j]
		den = h[i] + h[j]
		r_ff.append( 2 * num / den )
	r_ff = np.mean(r_ff)
	
	return r_ff

def f2c_entropy(X, y):
	y = y.reshape(len(y), 1)
	# create empty arrays
	joint_hy = np.zeros(X.shape[1]) - 1
	conditional_hy = np.zeros(X.shape[1]) - 1
	
	# compute entropy measures
	# H(Y)
	hy = continuous.get_h_mvn(y)
	for i in range(0, X.shape[1]):
		x = X[:, i].reshape(X.shape[0], 1)
		# H(X,Y)
		joint_hy[i] = continuous.get_mi_mvn(x, y)
		# H(X|Y) = H(X,Y) - H(Y)
		conditional_hy[i] = continuous.get_mi_mvn(x, y) - continuous.get_h_mvn(y)
	
	return hy, joint_hy, conditional_hy

def f2f_entropy(X):
	
	# create empty arrays
	h = np.zeros(X.shape[1]) - 1
	joint_h = np.zeros((X.shape[1], X.shape[1])) - 1
	conditional_h = np.zeros((X.shape[1], X.shape[1])) - 1
	
	# compute entropy measures
	for i in range(0, X.shape[1]):
		x1 = X[:, i].reshape(X.shape[0], 1)
		# H(X)
		h[i] = continuous.get_h_mvn(x1)
		for j in range(0, X.shape[1]):
			x2 = X[:, j].reshape(X.shape[0], 1)
			# H(X,Y)
			joint_h[i, j] = continuous.get_mi_mvn(x1, x2)
			# H(X|Y) = H(X,Y) - H(Y)
			conditional_h[i, j] = continuous.get_mi_mvn(x1, x2) - continuous.get_h_mvn(x2)
	
	return h, joint_h, conditional_h

if __name__ == "__main__" :
	
	i = 0
	errs = 0
	
	db_list = [
#					'wine', 
#					'diabetes', 
					'vehicle', 
#					'iris',
#					'hill-valley',
#					'arcene', 
#					'jasmine',
#					'dexter', 
#					'gisette', 
#					'dorothea', 
#					'madelon',
			]
	
	fs_methods = [
					'mi',
					'anova',
					'RFE',
					'EvoFS',
			]
	
#	import openml
	for key, db in openml.datasets.list_datasets().items():
#		if db['name'] in ['arcene']:
#			print(db)
#		try:
		if db['name'] in db_list:
#			if db['NumberOfInstances'] < 2000 and db['NumberOfFeatures'] < 1000:
		
		
			# create a new folder
			folder_name = './exhaustive/' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") + "-EvoFS-" + db['name']
			if not os.path.exists(folder_name) : 
				os.makedirs(folder_name)
			else :
				sys.stderr.write("Error: folder \"" + folder_name + "\" already exists. Aborting...\n")
				sys.exit(0)
		   
			for m in fs_methods:
				main(db['name'], 200, 200, db['did'], m, folder_name)
			
			db_list.remove(db['name'])
			
			i = i+1
#				break
			
#		except:
#			print("Error on database: " + str(db['did']))
#			errs = errs + 1
#		break
	sys.exit()
	

