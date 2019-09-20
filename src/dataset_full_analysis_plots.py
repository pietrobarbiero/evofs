import argparse
import datetime
import logging
import os
import sys

import matplotlib.pyplot as plt

from logging.handlers import RotatingFileHandler
from pandas import read_csv

def main() :
	
	# get command-line arguments
	args = parse_command_line()

	# initialize logging, using a logger that smartly manages disk occupation
	initialize_logging()

	# start program
	logging.info("Loading dataset \"%s\"..." % args.dataset)
	df = read_csv(args.dataset)
	columns = list(df)
	classifiers_columns = [ c for c in columns if c.endswith("_avg_test_accuracy") ]
	logging.info("Dataset loaded; found results for %d classifiers." % len(classifiers_columns))
	
	# prepare data structures to stock information
	best_feature_sets = dict()
	sizes = sorted( df['size'].unique() )
	
	for c in classifiers_columns :
		
		# initialize dict of best feature sets
		best_feature_sets[c] = dict()

		# identify the best feature set, for each size
		for s in sizes :
			df_selection = df[ df['size'] == s ]
			best_feature_set = df_selection.ix[ df_selection[c].idxmax() ]['feature_set'] # idxmax gives the row index for which column value is maximal
			best_feature_sets[c][s] = best_feature_set
		
		print("Best feature sets for " + c + ":", best_feature_sets[c])
	
	# and now, a second iteration on the classifiers, to have some plots
	for current_classifier in classifiers_columns :
		
		# first, let's check if stdev are included in the dataset
		stdev_column_name = current_classifier.split("_")[0] + "_stdev_test_accuracy"
		stdev_bars = []
		
		# collect all feature sets points (size, error)
		x = []
		y = []
		for index, row in df.iterrows() :
			# all the commented stuff is redundant :-D
			#is_row_best_for_any_classifier = False
			#
			#for c2 in classifiers_columns :
			#	if row['feature_set'] == best_feature_sets[c2][row['size']] :
			#		is_row_best_for_any_classifier = True
			#
			#if not is_row_best_for_any_classifier :
			x.append( int(row['size']) )
			y.append( 1.0 - float(row[current_classifier]) )
			if stdev_column_name in list(df) : stdev_bars.append( float(row[stdev_column_name]) )

		#print("x (size " + str(len(x)) + ") =", x)
		#print("y (size " + str(len(y)) + ") =", y)
		
		fig = plt.figure()
		ax = fig.add_subplot(111)

		if stdev_column_name in list(df) :	
			ax.errorbar(x, y, yerr=stdev_bars, fmt='.', color='black', label='exhaustive')
		else :
			ax.scatter(x, y, marker='+', color='black', label="exhaustive")
		
		for c2 in classifiers_columns :
			sizes = sorted( best_feature_sets[c2].keys() )
			# this seems complicated, but it's just: get the accuracy value for classifier 'current_classifier', 
			# in the row that contains the best feature set for classifier 'c2'
			accuracies = [ df[ df['feature_set'] == best_feature_sets[c2][s] ][current_classifier] for s in sizes ] 
			errors = [ float(1.0 - a) for a in accuracies ]

			# TODO with a lot of classifiers, plot in different colors
			marker = 'o'
			color = 'blue'
			if c2 == current_classifier : color = 'red' 
			
			ax.plot(sizes, errors, color=color, marker=marker, alpha=0.3, label="Best feature sets for " + c2)
		
		ax.set_xlabel("number of features")
		ax.set_ylabel("average test error (10-fold cv)")
		ax.set_title("Feature sets, %s" % current_classifier)
		ax.legend(loc='best')
		
		plt.savefig(current_classifier + ".png")
		plt.savefig(current_classifier + ".pdf")
		plt.show()
		plt.close(fig)

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
	
	parser = argparse.ArgumentParser(description="Plot stuff! <alberto.tonda@gmail.com>")
	
	# required argument
	parser.add_argument("-d", "--dataset", help="Dataset to be analyzed.", required=True)	
	
	# list of elements, type int
	#parser.add_argument("-rid", "--regionId", type=int, nargs='+', help="List of regional IDs. All SARs belonging to regions with these IDs will be included in the optimization process.")
	
	# flag, it's just true/false
	#parser.add_argument("-sz", "--startFromZero", action='store_true', help="If this flag is set, the algorithm will include an individual with genome '0' [0,0,...,0] in the initial population. Useful to improve speed of convergence, might cause premature convergence.")
		
	args = parser.parse_args()
	
	return args

if __name__ == "__main__" :
	sys.exit( main() )
