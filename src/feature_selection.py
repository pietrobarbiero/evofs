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
	if selectedDataset == 'madelon':
		X, y = datasets.make_classification(n_samples=4400,
											n_features=500,
											n_informative=5,
											n_redundant=15,
											shuffle=False,
											random_state=seed)
	else:
		db = openml.datasets.get_dataset(did)
		X, y, attribute_names = db.get_data(
				   target=db.default_target_attribute,
				      return_attribute_names=True)
		try:
			X = X.toarray()
		except:
			print("X was already a numpy array...")
#		X, y, categorical_indicator, attribute_names = db.get_data(
#		    dataset_format="array",
#		    target=db.default_target_attribute
#		)
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
		
		# print csv header
		with open( os.path.join(folder_name, 
						  "final_results_" + dbname + "_" + method + "_" + classifier_name + ".csv"), "w") as fp :
			fp.write("feature-set,size\n")
		
		# rescale data
		scaler = StandardScaler()
		sc = scaler.fit(X)
		X = sc.transform(X)
		
		print("Feature selection algorithm: %s" % (method))
		
		# standard methods
		if method in standard_methods:
			
			if method == 'mi': fs_class = mutual_info_classif
			elif method == 'anova': fs_class = f_classif
			elif method == 'RFE': fs_class = RFE
			
			feature_set, exec_time = standard_feature_selection(X, y, model, fs_class, dbname, classifier_name, method)
			
		elif method == 'EvoFS':
			
			initial_pop = []
			
			print("Computing ANOVA on the whole feature set...")
			# seed the initial population
#			feature_set_mi, exec_time = standard_feature_selection(X, y, model, mutual_info_classif, dbname, classifier_name, "mi")
			feature_set_anova, exec_time = standard_feature_selection(X, y, model, f_classif, dbname, classifier_name, "anova")
#			feature_set_RFE, exec_time = standard_feature_selection(X, y, model, RFE, dbname, classifier_name, "RFE")
			
#			initial_pop.extend(feature_set_mi)
			initial_pop.extend(feature_set_anova)
#			initial_pop.extend(feature_set_RFE)
			
			# we use the feature set from ANOVA to get a ordered list of importance of all features
			sorted_features = feature_set_anova[-1]
			print("Creating feature importance array...")
			feature_importance = np.zeros((len(sorted_features)))
			
			for i in range(0, len(sorted_features)) : 
				feature_importance[ sorted_features[i] ] = i
			
			csv_results_str = "final_results_" + dbname + "_" + method + "_" + classifier_name + ".csv"
			
			# start evolving feature selection
			start = time.time()
			final_archive = evolveFeatureSelection(X, y, 
											   model, pop_size, offspring_size, max_generations, 
											   initial_pop=initial_pop,
											   seed=seed, experiment_name=folder_name,
											   csv_results_str=csv_results_str,
											   feature_importance=feature_importance)
			end = time.time()
			exec_time = end - start
			
			final_archive = sorted(final_archive, key = lambda x : x.fitness[0])
			feature_set = []
			for solution in final_archive:
				sol = []
				j = 0
				for i in solution.candidate:
					if i == 1:
						sol.append(j)
					j = j + 1
				feature_set.append( tuple(sol) )
		
		with open( os.path.join(folder_name, 
						  "final_results_" + dbname + "_" + method + "_" + classifier_name + ".csv"), "a") as fp :
			for fs in feature_set :
				fp.write(str(fs).replace(","," ")) # replace ',' with ' '  in string representation of a list, avoid issues with CSV
				fp.write("," + str(len(fs)))
				fp.write("\n")
				
		with open( os.path.join(folder_name, 
						  "run_time_" + dbname + "_" + method + "_" + classifier_name + ".csv"), "w") as fp :
			fp.write(str(exec_time))

	return

# function that does most of the work
def evolveFeatureSelection(X, y,
						   classifier, pop_size, offspring_size, max_generations, 
						   initial_pop, seed=None, experiment_name=None,
						   csv_results_str=None,
						   feature_importance=None) :
	
	# initialize pseudo-random number generation
	prng = random.Random()
	prng.seed(seed)

	# set up MOEA
	print("\nSetting up evolutionary algorithm...")
	ea = inspyred.ec.emo.NSGA2(prng)
	ea.variator = [ variateFeatureSelection ]
	ea.terminator = inspyred.ec.terminators.generation_termination
	ea.observer = observeFeatureSelection
	#ea.replacer = replaceFeatureSelection # TODO comment/uncomment this line to activate/deactivate the experimental replacer
	maximize = False
	n_classes = len( np.unique(y) )
	
	# start evolution
	ea.evolve(    
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
					feature_importance = feature_importance,
					cross_validation_dictionary = dict(),
					n_classes = n_classes,
					classifier = classifier,
					X_train = X,
					y_train = y,
					experimentName = experiment_name,
					current_time = datetime.datetime.now(),
					numG = 0,
					seed = initial_pop,
					csv_results_str=csv_results_str)

	return ea.archive

# now, this is a new part of the program that attempts to modify the replacer
def replaceFeatureSelection(random, population, parents, offspring, args) :
	
	# retrieve useful variables
	cross_validation_dictionary = args.get("cross_validation_dictionary", None)
	
	# first, we run the replacer already prepared for NSGA-II, that should get the non-dominated individuals
	pareto_front = inspyred.ec.replacers.nsga_replacement(random, population, parents, offspring, args)
	
	# then, we find all discarded individuals
	discarded_individuals = [ i for i in population if i not in pareto_front ]
	
	# now, let's see how many discarded individuals might deserve a second chance;
	# they do, if we can find at least one individual on the non-dominated front of the same size, that is not separable
	# individual's accuracy values, for each fold in the K-fold, are stored in a dictionary
	second_chance = []
	for d in discarded_individuals :
		
		# go over comparable individuals
		comparable_individuals = [ i for i in discarded_individuals if i.fitness[0] == d.fitness[0] ]
		is_separable = True
		index = 0
		
		while index < len(comparable_individuals) and is_separable :
			
			# we get the accuracy for each fold
			d_values = cross_validation_dictionary[ str(d.genome) ]
			i_values = cross_validation_dictionary[ str(comparable_individuals[index].genome) ]
			
			# we compare them
			t, p = stats.ttest_ind(d_values, i_values, equal_var=False)
			if p > 0.05 : is_separable = False
		
			index += 1
		
		if not is_separable : second_chance.append( d )
		
	pareto_front.extend( second_chance )
	return pareto_front 

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
	feature_importance = args.get("feature_importance", None)
	
	seed = 42
	index = 1
	fitness = []
	
	# compute the fitness of each individual
	for c in candidates :
		
		cAsBoolArray = np.array(c, dtype=bool)
		
		# compute the number of selected features
		nSelectedFeatures = np.sum( cAsBoolArray )
			
		# compute individual feature importance
		#F, pval = f_classif(X[:, cAsBoolArray], y)
		#statTest = -np.min(F)
		
		# REPLACED by using the pre-computed feature importance array
		# in the feature importance array, each feature index is associated with its own order of importance
		# so, higher order of importance is worse; as we are trying to minimize everything, it should be ok
		# to take the max feature importance order in the subset
		feature_importance_subset = feature_importance[cAsBoolArray]
		statTest = np.max(feature_importance_subset)
		
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
		
		fitness.append( inspyred.ec.emo.Pareto( [nSelectedFeatures, error, statTest] ) )
		
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
	
	print("[%s] Generation %d, Best individual: selected features=%.2f, error=%.4f, F=%.2f" 
				   % (delta_time_string, num_generations, pop[0].fitness[0], pop[0].fitness[1], pop[0].fitness[2]))
	
#	print("[%s] Generation %d, Best individual: selected features=%.2f, CFS=%.4f" 
#				   % (delta_time_string, num_generations, pop[0].fitness[0], pop[0].fitness[1]))
	
	args["current_time"] = current_time
	args["numG"] = num_generations
	
	
	# save best individuals
	final_archive = args["_ec"].archive
	csv_results_str = args["csv_results_str"]
	if final_archive != None :
		
		# print csv header
		with open( os.path.join(folder_name, csv_results_str), "w") as fp :
			fp.write("feature-set,size\n")
		
		# sort solutions
		final_archive = sorted(final_archive, key = lambda x : x.fitness[0])
		feature_set = []
		for solution in final_archive:
			sol = []
			j = 0
			for i in solution.candidate:
				if i == 1:
					sol.append(j)
				j = j + 1
			feature_set.append( tuple(sol) )
		
		# write solutions
		with open( os.path.join(folder_name, csv_results_str), "a") as fp :
			for fs in feature_set :
				fp.write(str(fs).replace(","," ")) # replace ',' with ' '  in string representation of a list, avoid issues with CSV
				fp.write("," + str(len(fs)))
				fp.write("\n")

	return

def standard_feature_selection(X, y, model, fs_algorithm, dbname, classifier_name, method):
	
	feature_set = []
	k = 1
	
	start = time.time()
		
	if method == 'RFE':
		referenceClassifier = copy.deepcopy(model)
		fs = RFE(referenceClassifier, k, step=1)
		fs.fit(X, y)
		scores = fs.ranking_
		rank = np.argsort(scores)
	else:
		fs = SelectKBest(fs_algorithm, k=k)
		fs.fit(X, y)
		scores = fs.scores_
		rank = np.argsort(-scores)
	
	fset = []
	for f in rank:
		fset.append(f)
		feature_set.append( tuple(fset) )
		
	end = time.time()
	exec_time = end - start
	
	return feature_set, exec_time

if __name__ == "__main__" :
	
	i = 0
	errs = 0
	
	db_list = [
#					'wine', 
#					'Australian',
#					'diabetes', 
#					'vehicle',
#					'iris', 
#					'madelon',
#					'hill-valley',
#					'arcene', 
#					'jasmine',
#					'Dexter', 
					'gisette', 
#					'dorothea',
			]
	
	fs_methods = [
#					'anova',
#					'mi',
#					'RFE',
					'EvoFS',
			]
	
#	import openml
	for key, db in openml.datasets.list_datasets().items():
#		if db['name'] == 'gisette':
#			print(1)
		
		# io voglio tanto bene a chi ha scritto questo codice, ma la prossima volta che becco un except "everything"
		# che MASCHERA I MESSAGGI DI ERRORE, ammazzo qualcuno
		#try:
			
		if db['name'] in db_list:
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
		
		#except Exception as e:
		#	print("Error on database " + str(db['did']) + ":\"" + str(e) + "\"")
		#	errs = errs + 1
#		break
	sys.exit()
	

