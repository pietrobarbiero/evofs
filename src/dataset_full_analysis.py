# Script performing an 'exhaustive' analysis of the feature selection process in a dataset
import argparse
import copy
import datetime
import itertools
import logging
import multiprocessing # attempt to parallelize heavy computations
import numpy as np
import os
import random
import sys

# scikit-learn stuff
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# logging module extras
from logging.handlers import RotatingFileHandler

# pandas
from pandas import read_csv

# local libraries
import fitness

def main() :
	
	# default and hard-coded values
	#dataset_name = 'wine'
	dataset_name = 'vehicle'
	random_seed = 42
	n_splits = 10
	
	# get command-line arguments
	args = parse_command_line()
	if args.dataset : dataset_name = args.dataset
	
	# name of the final file depends on the dataset
	analysis_file = "final_results_" + dataset_name + ".csv"

	# create folder with unique name
	folderName = ""
	if args.folder :
		folderName = args.folder
	else :
		folderName = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")  
		folderName += "-exhaustive-analysis"
		if not os.path.exists(folderName) : os.makedirs(folderName)

	# initialize logging, using a logger that smartly manages disk occupation
	initialize_logging(folderName)

	feature_sets_already_analyzed = []
	if args.folder :
		# load dataset
		logging.info("Loading existing file \"%s\"..." % os.path.join(folderName, analysis_file))
		df = read_csv( os.path.join(folderName, analysis_file) ) 
		feature_sets_already_analyzed = df['feature_set'].values
		# TODO: find a way to store/restore last known random state

	# start program
	logging.info("Attempting to fetch dataset \"%s\"..." % dataset_name)
	X, y, feature_names = fetch_dataset(dataset_name)
	
	# some pre-processing, normalize features and get a list of folds
	logging.info("Pre-processing data...")
	from sklearn.preprocessing import StandardScaler
	x_scaler = StandardScaler()
	X = x_scaler.fit_transform(X)
	
	from sklearn.model_selection import StratifiedKFold
	skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
	folds = [ [train_index, test_index] for train_index, test_index in skf.split(X,y) ]
	
	# this is an additional part, where we try to get A LOT of extra folds
	#random.seed(random_seed)
	#for i in range(0, 9) : # 9 * n_splits additional splits
	#	skf2 = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random.randint(0, 99999))
	#	folds.extend( [ [train_index, test_index] for train_index, test_index in skf2.split(X,y) ] )
	
	logging.info("A total of %d folds will be used." % len(folds))
	
	# get all possible features subsets
	logging.info("Computing all subsets of the %d features..." % len(feature_names))
	features_indexes = [ i for i in range(0, len(feature_names)) ]
	features_sets = powerset(features_indexes)
	logging.info("Found a total of %d subsets" % len(features_sets))
	
	# data structures to save results
	performance = dict()
	classifiers = [ 
#			RandomForestClassifier(), 
			LogisticRegression(solver='lbfgs', multi_class='auto') 
			] 
	
	for fs in features_sets : performance[fs] = dict()
	
	# initialize header of output file
	if not os.path.exists( os.path.join(folderName, analysis_file) ) :
		with open( os.path.join(folderName, analysis_file), "w" ) as fp :
			# header
			fp.write("feature_set,size")
			# store average and stdev of the accuracy for all classifiers
			for classifier in classifiers : 
				fp.write("," + classifier.__class__.__name__ + "_avg_test_accuracy")
				fp.write("," + classifier.__class__.__name__ + "_stdev_test_accuracy")
			# also store the result of each fold, separately
			for classifier in classifiers :
				for i in range(0, len(folds)) :
					fp.write("," + classifier.__class__.__name__ + "accuracy_fold_" + str(i))
			fp.write("\n")
	
	# TODO REMOVE THIS LINE, IT REVERSES THE LIST
	features_sets.reverse()
	
	# evaluate all possible subsets
	#for fs_index, fs in enumerate(features_sets[:10]) : # debug
	for fs_index, fs in enumerate(features_sets) :
		
		logging.info("Analyzing feature subset %d/%d, of size %d..." % (fs_index, len(features_sets), len(fs)))

		# check if fs is inside the list of feature sets already processed
		if str(fs).replace(","," ") not in feature_sets_already_analyzed :
			logging.info("")
			X_reduced = X[:,fs]
			logging.info("Training file will be of size %s" % (str(X_reduced.shape)))
			
			# perform a n-fold cross-validation using different classifiers, record all test accuracy
			for classifier in classifiers :
				logging.info(	"Performing a %d-fold cross-validation (%d folds) with classifier \"%s\"..." % 
						(n_splits, len(folds), classifier.__class__.__name__))
				performance[fs][ classifier.__class__.__name__ ] = fitness.cross_validation(folds, X_reduced, y, classifier) #, return_error=True) # TODO set the flag to store error instead of classificationaccuracy
			
			# now, all values are immediately written to file instead of waiting at the end
			# slower, but it makes it possible to resume an analysis
			with open( os.path.join(folderName, analysis_file), "a") as fp :
				fp.write(str(fs).replace(","," ")) # replace ',' with ' '  in string representation of a list, avoid issues with CSV
				fp.write("," + str(len(fs)))
				# average and stdev of the accuracy
				for classifier in classifiers :
					logging.debug("All results for feature set %s, classifier %s: %s" %
							(str(fs), classifier.__class__.__name__, str(performance[fs][classifier.__class__.__name__])))
					avg_test_accuracy = np.average( performance[fs][classifier.__class__.__name__] )
					stdev_test_accuracy = np.std( performance[fs][classifier.__class__.__name__] )
					fp.write("," + str(avg_test_accuracy))
					fp.write("," + str(stdev_test_accuracy))
				# accuracy for each fold
				for classifier in classifiers :
					for result in performance[fs][classifier.__class__.__name__] :
						fp.write("," + str(result))
				fp.write("\n")
		else :
			logging.info("Feature set already found, skipping...")
		
		# TODO compute Fontanella's fitness value
	
	# save final results
#	with open( os.path.join(folderName, analysis_file), "w") as fp :
#
#		# header
#		fp.write("feature_set,size")
#		# store average and stdev of the accuracy for all classifiers
#		for classifier in classifiers : 
#			fp.write("," + classifier.__class__.__name__ + "_avg_test_accuracy")
#			fp.write("," + classifier.__class__.__name__ + "_stdev_test_accuracy")
#		# also store the result of each fold, separately
#		for classifier in classifiers :
#			for i in range(0, len(folds)) :
#				fp.write("," + classifier.__class__.__name__ + "accuracy_fold_" + str(i))
#		fp.write("\n")
#
#		#for fs in features_sets[:10] : # debug
#		for fs in features_sets :
#			fp.write(str(fs).replace(","," ")) # replace ',' with ' '  in string representation of a list, avoid issues with CSV
#			fp.write("," + str(len(fs)))
#			# average and stdev of the accuracy
#			for classifier in classifiers :
#				logging.debug("All results for feature set %s, classifier %s: %s" %
#						(str(fs), classifier.__class__.__name__, str(performance[fs][classifier.__class__.__name__])))
#				avg_test_accuracy = np.average( performance[fs][classifier.__class__.__name__] )
#				stdev_test_accuracy = np.std( performance[fs][classifier.__class__.__name__] )
#				fp.write("," + str(avg_test_accuracy))
#				fp.write("," + str(stdev_test_accuracy))
#			# accuracy for each fold
#			for classifier in classifiers :
#				for result in performance[fs][classifier.__class__.__name__] :
#					fp.write("," + str(result))
#			fp.write("\n")
#		
#		# TODO store all other metrics related to the feature set

	return

# finding all subsets of all sizes from a list
def powerset(iterable):
	"powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3) : no empty sets considered"
	s = list(iterable)
	return list( itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1, len(s)+1)) )

# attempts to find a dataset in the OpenML repository
def fetch_dataset(name) :
	
	import openml
	X = y = feature_names = None

	# TODO error control, try...catch
	for key, db in openml.datasets.list_datasets().items():
		if db['name'] == name and X is None :
			db = openml.datasets.get_dataset(db['did'])
			X, y, feature_names = db.get_data(target=db.default_target_attribute, return_attribute_names=True)
	
	if X is None : logging.warning("Dataset %s not found!" % name)
	
	return X, y, feature_names

def initialize_logging(folderName=None) :
	logger = logging.getLogger()
	logger.setLevel(logging.DEBUG)
	formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(message)s') 

	# the 'RotatingFileHandler' object implements a log file that is automatically limited in size
	if folderName != None :
		fh = RotatingFileHandler( os.path.join(folderName, "log.log"), mode='a', maxBytes=100*1024*1024, backupCount=2, encoding=None, delay=0 )
		fh.setLevel(logging.DEBUG)
		fh.setFormatter(formatter)
		logger.addHandler(fh)

	ch = logging.StreamHandler()
	ch.setLevel(logging.INFO)
	ch.setFormatter(formatter)
	logger.addHandler(ch)
	
	return

def parse_command_line() :
	
	parser = argparse.ArgumentParser(description="Python script performing an 'exhaustive' analysis of the feature selection in a dataset. Pietro Barbiero and Alberto Tonda, 2019 <alberto.tonda@gmail.com>")
	
	# required argument
	#parser.add_argument("-sar", "--sar", help="File containing the list of all the SARs (Small Agricultural Regions).", required=True)	
	parser.add_argument("-d", "--dataset", help="Name of the OpenML dataset that will be analyzed")
	
	# not required, number of max parallel processes
	parser.add_argument("-p", "--processes", type=int, help="Maximum number of parallel processes to run (default: all available resources, minus two)")
	
	# not required, check whether to resume from a folder
	parser.add_argument("-f", "--folder", help="If specified, looks for the output file inside the folder, and resumes the exhaustive analysis from the last entries in that file. Used to stop and restart long analyses.")

	# list of elements, type int
	#parser.add_argument("-rid", "--regionId", type=int, nargs='+', help="List of regional IDs. All SARs belonging to regions with these IDs will be included in the optimization process.")
	
	# flag, it's just true/false
	#parser.add_argument("-sz", "--startFromZero", action='store_true', help="If this flag is set, the algorithm will include an individual with genome '0' [0,0,...,0] in the initial population. Useful to improve speed of convergence, might cause premature convergence.")
		
	args = parser.parse_args()
	
	return args

if __name__ == "__main__" :
	sys.exit( main() )
