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
#			[BaggingClassifier, "BaggingClassifier", 1],
#			[SVC, "SVC", 1],
			[RidgeClassifier, "RidgeClassifier", 1],
			]
	
	selectedClassifiers = [classifier[1] for classifier in allClassifiers]
	
	# load different datasets, prepare them for use
	print("Preparing data...")
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
	print("Settings of the experiment...")
	print("Fixed random seed:", seed)
	print("Selected dataset: %s; Selected classifier(s): %s" % (selectedDataset, selectedClassifiers))
	print("Population size in EA: %d; Offspring size: %d; Max generations: %d" % (pop_size, offspring_size, max_generations))

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
		
		# create splits for cross-validation
		print("Creating train/test split...")
		skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
		
		split_index = 1
		for train_index, test_index in skf.split(X, y) :
		
			X_train, y_train = X[train_index], y[train_index]
			X_test, y_test = X[test_index], y[test_index]
			print("\tSplit %d" %(split_index))
			
			# rescale data
			scaler = StandardScaler()
			sc = scaler.fit(X_train)
			X = sc.transform(X)
			X_train = sc.transform(X_train)
			X_test = sc.transform(X_test)
			
			# standard methods
			if method in standard_methods:
				
				if method == 'mi': fs_class = mutual_info_classif
				elif method == 'anova': fs_class = f_classif
				elif method == 'RFE': fs_class = RFE
				
				standard_feature_selection(X_train, y_train, X_test, y_test, 
							   model, fs_class, dbname, classifier_name, method)
			
			elif method == 'exhaustive':
				
				exhaustive_search(X_train, y_train, X_test, y_test, model, dbname, classifier_name, method)
				
			elif method == 'EvoFS':
				
				# seed the initial population using MI
				initial_pop = []
				for k in range(1, X.shape[1]+1):
					# MI individuals
					fs_mi = SelectKBest(mutual_info_classif, k=k)
					fs_mi.fit(X_train, y_train)
					# RFE individuals
					referenceClassifier = copy.deepcopy(model)
					fs_rfe = RFE(referenceClassifier, k, step=1)
					fs_rfe.fit(X_train, y_train)
					# ANOVA individuals
					fs_anova = SelectKBest(f_classif, k=k)
					fs_anova.fit(X_train, y_train)
					
					initial_pop.append( list( fs_mi.get_support() ) )
					initial_pop.append( list( fs_rfe.get_support() ) )
					initial_pop.append( list( fs_anova.get_support() ) )
				
				# start evolving feature selection
				start = time.time()
				final_archive, trainAccuracy, testAccuracy = evolveFeatureSelection(X_train, y_train, X_test, y_test, 
												   model, pop_size, offspring_size, max_generations, 
												   initial_pop=initial_pop,
												   seed=seed, experiment_name=folder_name)
				
				feature_set = list(set([ f.fitness[0] for f in final_archive ]))
				best_solutions = []
				train_error = []
				test_error = []
				for fs in feature_set:
					solutions = [f for f in final_archive if f.fitness[0] == fs]
					solutions = sorted(solutions, key = lambda x : 1-x.fitness[1])
					best_solutions.append(solutions[-1])
					# retain only the features selected according to the candidate solution
					cAsBoolArray = np.array(solutions[-1].candidate, dtype=bool)
					X_train_reduced = X_train[:, cAsBoolArray]
					X_test_reduced = X_test[:, cAsBoolArray]
					# compute classification accuracy
					referenceClassifier = copy.deepcopy(model)
					referenceClassifier.fit(X_train_reduced, y_train)
					accuracy_train = referenceClassifier.score( X_train_reduced, y_train )
					train_error.append( 1-accuracy_train )
					accuracy = referenceClassifier.score( X_test_reduced, y_test )
					test_error.append( 1-accuracy )
					
				pareto_front_x = [ f.fitness[0] for f in best_solutions ]
				
				end = time.time()
				exec_time = end - start
				
				with open( os.path.join(folder_name, "%s__%s_%s_train.csv" 
							%(dbname, classifier_name, method)), "w", newline='') as fp:
					wr = csv.writer(fp)
					wr.writerow( len(pareto_front_x)*[exec_time] )
					wr.writerow( pareto_front_x )
					wr.writerow( train_error )
					
				with open( os.path.join(folder_name, "%s__%s_%s_test.csv" 
							%(dbname, classifier_name, method)), "w", newline='') as fp:
					wr = csv.writer(fp)
					wr.writerow( len(pareto_front_x)*[exec_time] )
					wr.writerow( pareto_front_x )
					wr.writerow( test_error )
			
			split_index = split_index + 1
			
			break

	return

# function that does most of the work
def evolveFeatureSelection(X_train, y_train, X_test, y_test, 
						   classifier, pop_size, offspring_size, max_generations, 
						   initial_pop, seed=None, experiment_name=None) :
	
	# initialize pseudo-random number generation
	prng = random.Random()
	prng.seed(seed)

	# set the reference performance as the one using all features
	print("Computing initial classifier performance using all features...")
	referenceClassifier = copy.deepcopy(classifier)
	referenceClassifier.fit(X_train, y_train)
	y_train_pred = referenceClassifier.predict(X_train)
	y_test_pred = referenceClassifier.predict(X_test)
	trainAccuracy = accuracy_score(y_train, y_train_pred)
	testAccuracy = accuracy_score(y_test, y_test_pred)
	print("Initial performance: train=%.4f, test=%.4f" % (trainAccuracy, testAccuracy))

	# set up MOEA
	print("\nSetting up evolutionary algorithm...")
	ea = inspyred.ec.emo.NSGA2(prng)
	ea.variator = [ variateFeatureSelection ]
	ea.terminator = inspyred.ec.terminators.generation_termination
	ea.observer = observeFeatureSelection
	maximize = False
	n_classes = len( np.unique(y_train) )
	
	# compute useful entropy measures
	hy, joint_hy, conditional_hy = f2c_entropy(X_train, y_train)
	h, joint_h, conditional_h = f2f_entropy(X_train)
	
	# start evolution
	final_population = ea.evolve(    
					generator = generateFeatureSelection,
					
#					evaluator = evaluateFeatureSelection,
					
					# this part is defined to use multi-process evaluations
					evaluator = inspyred.ec.evaluators.parallel_evaluation_mp,
					mp_evaluator = evaluateFeatureSelection, 
					mp_num_cpus = multiprocessing.cpu_count()-2,
					
					pop_size = np.max([ len(initial_pop), pop_size ]),
					num_selected = offspring_size,
					maximize = maximize, 
					max_generations = max_generations,
					
					# extra arguments here
					n_classes = n_classes,
					classifier = classifier,
					X_train = X_train,
					y_train = y_train,
					experimentName = experiment_name,
					current_time = datetime.datetime.now(),
					numG = 0,
					seed = initial_pop,
					# entropy measures
					hy = hy, 
					joint_hy = joint_hy, 
					conditional_hy = conditional_hy,
					h = h, 
					joint_h = joint_h, 
					conditional_h = conditional_h)

	return ea.archive, trainAccuracy, testAccuracy

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
	h = args.get("h", None)
	joint_h = args.get("joint_h", None)
	conditional_h = args.get("conditional_h", None)
	hy = args.get("hy", None)
	joint_hy = args.get("joint_hy", None)
	conditional_hy = args.get("conditional_hy", None)
	
	seed = 42
	index = 1
	fitness = []
	
	# compute the fitness of each individual
	for c in candidates :
		
		cAsBoolArray = np.array(c, dtype=bool)
		
		# compute the number of selected features
		nSelectedFeatures = np.sum( cAsBoolArray )
		
		if True:
			
			# compute individual feature importance
			mi = mutual_info_classif(X, y)
			mi[ np.isnan(mi)==True ] = 0
			mi[ np.isinf(mi)==True ] = 0
			statTest = 1-np.average(mi, weights=mi)
			
#			if nSelectedFeatures > 1:
#				positions = np.argwhere(cAsBoolArray==1).squeeze().tolist()
#				k = len(positions)
#				
#				# compute the average feature-feature correlation
#				rFF = r_ff(positions, h, joint_h, conditional_h)
#				# compute the average feature-class correlation
#				rCF = r_cf(positions, h, hy, joint_hy, conditional_h)
#				
#				# compute the (negative as we are minimizing) correlation-based feature selection fitness
#				statTest = - cfs_fitness(k, rFF, rCF)
#			else:
#				# horrible fitness
#				statTest = 0
				
			
			# split the data into training and validation sets
			skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=num_generations*index)
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
#			
		else:
		
			if nSelectedFeatures > 1:
				positions = np.argwhere(cAsBoolArray==1).squeeze().tolist()
				k = len(positions)
				
				# compute the average feature-feature correlation
				rFF = r_ff(positions, h, joint_h, conditional_h)
				# compute the average feature-class correlation
				rCF = r_cf(positions, h, hy, joint_hy, conditional_h)
				
				# compute the (negative as we are minimizing) correlation-based feature selection fitness
				cfs = - cfs_fitness(k, rFF, rCF)
			else:
				# horrible fitness
				cfs = 0
		
		fitness.append( inspyred.ec.emo.Pareto( [nSelectedFeatures, error, statTest] ) )
#		fitness.append( inspyred.ec.emo.Pareto( [nSelectedFeatures, cfs] ) )
		
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
	
	print("[%s] Generation %d, Best individual: selected features=%.2f, error=%.4f, MI=%.2f" 
				   % (delta_time_string, num_generations, pop[0].fitness[0], pop[0].fitness[1], 1-pop[0].fitness[2]))
	
#	print("[%s] Generation %d, Best individual: selected features=%.2f, CFS=%.4f" 
#				   % (delta_time_string, num_generations, pop[0].fitness[0], pop[0].fitness[1]))
	
	args["current_time"] = current_time
	args["numG"] = num_generations

	return

def standard_feature_selection(X_train, y_train, X_test, y_test, model, fs_algorithm, dbname, classifier_name, method):
	
	train_error = []
	test_error = []
	n_fs = []
	
	start = time.time()
	for k in range(1, X_train.shape[1]+1):
		
		if method == 'RFE':
			referenceClassifier = copy.deepcopy(model)
			fs = RFE(referenceClassifier, k, step=1)
			X_train_reduced = fs.fit_transform(X_train, y_train)
			X_test_reduced = fs.transform(X_test)
		else:
			fs = SelectKBest(fs_algorithm, k=k)
			X_train_reduced = fs.fit_transform(X_train, y_train)
			X_test_reduced = fs.transform(X_test)
		n_fs.append(X_train_reduced.shape[1])
		
		# compute classification accuracy using the selected features
		referenceClassifier = copy.deepcopy(model)
		referenceClassifier.fit(X_train_reduced, y_train)
		accuracy_train = referenceClassifier.score( X_train_reduced, y_train )
		train_error.append( 1-accuracy_train )
		accuracy_test = referenceClassifier.score( X_test_reduced, y_test )
		test_error.append( 1-accuracy_test )
	end = time.time()
	exec_time = end - start
	
	with open( os.path.join(folder_name, "%s__%s_%s_train.csv" 
				%(dbname, classifier_name, method)), "w", newline='') as fp:
		wr = csv.writer(fp)
		wr.writerow( len(n_fs)*[exec_time] )
		wr.writerow( n_fs )
		wr.writerow( train_error )
	
	with open( os.path.join(folder_name, "%s__%s_%s_test.csv" 
				%(dbname, classifier_name, method)), "w", newline='') as fp:
		wr = csv.writer(fp)
		wr.writerow( len(n_fs)*[exec_time] )
		wr.writerow( n_fs )
		wr.writerow( test_error )
	
	return


def exhaustive_search(X_train, y_train, X_test, y_test, model, dbname, classifier_name, method):
	
	train_error = []
	test_error = []
	n_fs = []
	fs_set = np.arange(0, X_train.shape[1])
	
	start = time.time()
	for k in range(1, X_train.shape[1]+1):
		
		# find all possible k-combinations of features
		fs_list = list( combinations(fs_set, k) )
		
		for f in fs_list:
			X_train_reduced = X_train[:, f]
			X_test_reduced = X_test[:, f]
			
			# compute classification accuracy using the selected features
			referenceClassifier = copy.deepcopy(model)
			referenceClassifier.fit(X_train_reduced, y_train)
			accuracy_train = referenceClassifier.score( X_train_reduced, y_train )
			train_error.append( 1-accuracy_train )
			accuracy_test = referenceClassifier.score( X_test_reduced, y_test )
			test_error.append( 1-accuracy_test )
			n_fs.append( len(f) )
			
	end = time.time()
	exec_time = end - start
	
	with open( os.path.join(folder_name, "%s__%s_%s_train.csv" 
				%(dbname, classifier_name, method)), "w", newline='') as fp:
		wr = csv.writer(fp)
		wr.writerow( len(n_fs)*[exec_time] )
		wr.writerow( n_fs )
		wr.writerow( train_error )
	
	with open( os.path.join(folder_name, "%s__%s_%s_test.csv" 
				%(dbname, classifier_name, method)), "w", newline='') as fp:
		wr = csv.writer(fp)
		wr.writerow( len(n_fs)*[exec_time] )
		wr.writerow( n_fs )
		wr.writerow( test_error )
		
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
					'EvoFS',
					'exhaustive',
					'mi',
					'anova',
					'RFE',
			]
	
#	import openml
	for key, db in openml.datasets.list_datasets().items():
#		if db['name'] in ['arcene']:
#			print(db)
#		try:
		if db['name'] in db_list:
#			if db['NumberOfInstances'] < 2000 and db['NumberOfFeatures'] < 1000:
		
		
			# create a new folder
			folder_name = './old/' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") + "-EvoFS-" + db['name']
			if not os.path.exists(folder_name) : 
				os.makedirs(folder_name)
			else :
				sys.stderr.write("Error: folder \"" + folder_name + "\" already exists. Aborting...\n")
				sys.exit(0)
		   
			for m in fs_methods:
				main(db['name'], 100, 50, db['did'], m, folder_name)
			
			db_list.remove(db['name'])
			
			i = i+1
#				break
			
#		except:
#			print("Error on database: " + str(db['did']))
#			errs = errs + 1
#		break
	sys.exit()
	

