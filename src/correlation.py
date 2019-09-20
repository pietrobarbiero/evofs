# This script tries to evaluate the correlation between different metrics used to assess the effectiveness of a feature subset
# by Pietro Barbiero and Alberto Tonda, 2019 <alberto.tonda@gmail.com>

import itertools
import matplotlib.pyplot as plt
import numpy as np
import openml
import pandas as pd
import sys

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.metrics import r2_score

def main() :
	
	dataset_results = "../results/wine_exhaustive_logisticRegression_randomForest_100_folds.csv"
	dataset_name = 'wine'

	print("Obtaining dataset details from OpenML...")
	dataset_id = get_dataset_id(dataset_name)
	dataset = openml.datasets.get_dataset(dataset_id)
	
	print("Loading dataset results \"%s\"..." % dataset_results)
	df = pd.read_csv(dataset_results)
	
	# let's try to visualize the correlation of a few scores
	scores = dict()
	scores["Random Forest average accuracy"] = df["RandomForestClassifier_avg_test_accuracy"]
	scores["Logistic Regression average accuracy"] = df["LogisticRegression_avg_test_accuracy"]
	print("Computing Mutual Information scores...")
	scores["Mutual Information"] = compute_mutual_information(dataset, df["feature_set"], function=mutual_info_classif)
	scores["ANOVA"] = compute_mutual_information(dataset, df["feature_set"], function=f_classif)
	# TODO compute and add some more scores here
	
	# and now, several plots with all the tuples of different scores
	list_of_scores = sorted( scores.keys() )
	
	for i in range(0, len(list_of_scores)) :
		for j in range(i+1, len(list_of_scores)) :
		
			score1_name = list_of_scores[i]
			score2_name = list_of_scores[j]
			print("Now analyzing \"%s\" vs \"%s\"..." % (score1_name, score2_name))
			
			score1 = scores[score1_name]
			score2 = scores[score2_name]

			r2 = r2_score(score1, score2)
			
			fig = plt.figure()
			ax = fig.add_subplot(111)
			ax.scatter(score1, score2, marker='o', alpha=0.3, label=None)
			ax.plot([min(score1), 1.0], [min(score1), 1.0], linestyle='--', color='red', label='1:1')
			
			ax.set_title("Comparison between metrics, R2=%.4f" % r2)
			ax.set_xlabel(score1_name)
			ax.set_ylabel(score2_name)
			ax.legend(loc='best')
			
			figure_name = score1_name.replace(" ", "") + "_vs_" + score2_name.replace(" ", "") 
			plt.savefig(figure_name + ".pdf")
			plt.savefig(figure_name + ".png")
			plt.close(fig)

	return

def get_dataset_id(dataset_name) :
	
	for key, db in openml.datasets.list_datasets().items() :
		if db['name'] == dataset_name :
			return db['did']
	
	return None

def compute_mutual_information(dataset, feature_subsets, function=mutual_info_classif) :
	
	scores = []
	X, y, feature_names = dataset.get_data(target=dataset.default_target_attribute, return_attribute_names=True)
	
	for feature_subset in feature_subsets :
		
		# get list of integers from string
		features_indexes = [ int(f) for f in feature_subset[1:-1].split(" ") if len(f) > 0 ]
		#print(features_indexes)
		
		X_reduced = X[:, features_indexes]
		mi_score = function(X_reduced, y)
		print(mi_score)
		
		scores.append( np.mean(mi_score) )
	
	return np.array(scores)

if __name__ == "__main__" :
	sys.exit( main() )
