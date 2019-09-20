import argparse
import datetime
import logging
import os
import sys

from logging.handlers import RotatingFileHandler
from pandas import read_csv
from scipy import stats

def main() :
	
	# hard-coded values
	column_name_avg = "_avg_test_accuracy"
	column_name_std = "_stdev_test_accuracy"
	column_name_fold = "_fold_"
	significance_threshold = 0.05 # significance threshold for the statistical tests (e.g. "p < 0.05")
	
	# get command-line arguments
	args = parse_command_line()

	# create folder with unique name
	folderName = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")  
	folderName += "-statistical-analysis"
	if not os.path.exists(folderName) : os.makedirs(folderName)

	# initialize logging, using a logger that smartly manages disk occupation
	initialize_logging(folderName)

	# start program
	logging.info("Reading dataset \"%s\"..." % args.dataset)
	df = read_csv(args.dataset)
	
	# find all column names for each classifier
	classifier_names = [ c.split('_')[0] for c in list(df) if c.endswith("_avg_test_accuracy") ]
	logging.info("Found %d different classifiers:" % len(classifier_names))
	for c in classifier_names : logging.info("- \"%s\"" % c)
	
	# get some information from the dataset
	sizes = sorted( [ int(s) for s in df['size'].unique() ] )
	
	# so, we are interested in performing some statistical tests, to see if the averages are reliable
	# first test: for each classifier, for each feature set size, is the best feature set separable from the others?
	for c in classifier_names :
		logging.info("---")
		logging.info("Performing first test for \"%s\"..." % c)
		
		columns_folds = [ c for c in list(df) if column_name_fold in c ]
		logging.debug("Names of fold columns: %s", str(columns_folds))
		
		for s in sizes :
			non_separable_sets_t = 0
			non_separable_sets_ks = 0

			df_selection = df[ df['size'] == s ]
			logging.info("There are %d feature sets for size %d" % (len(df_selection), s))
			
			# find the row with the best feature set (by test accuracy average)
			best_row = df_selection.ix[ df_selection[c + column_name_avg].idxmax() ]
			best_set = best_row['feature_set']
			best_scores = [ best_row[f] for f in columns_folds ]
			logging.info("The best feature set for this size is %s" % str(best_set))
			logging.debug("With scores: %s", str(best_scores))
			
			# and now, let's iterate over the lines of the selection!
			for row_index, row in df_selection.iterrows() :
				
				if row['feature_set'] != best_set :
					logging.info("Comparing feature set %d with best feature set of size %d for %s..." % (row_index, s, c))
					
					current_scores = [ row[f] for f in columns_folds ]
					t, p = stats.ttest_ind(best_scores, current_scores)
					logging.info("Using T-test, the results are different with p=%.4f" % p)
					if p >= significance_threshold : non_separable_sets_t += 1
					
					d, p = stats.ks_2samp(best_scores, current_scores)
					logging.info("Using KS, the results are different with p=%.4f" % p)
					if p >= significance_threshold : non_separable_sets_ks += 1
					
			logging.info("Summary: using the T-test, for size %d, %d/%d feature sets were not separable from the best." % 
					(s, non_separable_sets_t, len(df_selection)))
			logging.info("Summary: using the KS-test, for size %d, %d/%d feature sets were not separable from the best." % 
					(s, non_separable_sets_ks, len(df_selection)))
		

	return

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
	
	parser = argparse.ArgumentParser(description="Python script to perform statistical analysis on datasets. 2019 <alberto.tonda@gmail.com>")
	
	# required argument
	parser.add_argument("-d", "--dataset", help="CSV file containing the data. Has to be formatted in a specific way.", required=True)	
	
	# list of elements, type int
	#parser.add_argument("-rid", "--regionId", type=int, nargs='+', help="List of regional IDs. All SARs belonging to regions with these IDs will be included in the optimization process.")
	
	# flag, it's just true/false
	#parser.add_argument("-sz", "--startFromZero", action='store_true', help="If this flag is set, the algorithm will include an individual with genome '0' [0,0,...,0] in the initial population. Useful to improve speed of convergence, might cause premature convergence.")
		
	args = parser.parse_args()
	
	return args

if __name__ == "__main__" :
	sys.exit( main() )
