import argparse
import datetime
import logging
import os
import sys

import matplotlib.pyplot as plt

from logging.handlers import RotatingFileHandler
from pandas import read_csv
import pandas as pd
import numpy as np
from ast import literal_eval
import openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import copy
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from scipy import stats
from sklearn import datasets


import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.rcParams.update({'font.size': 16})


#def initialize_logging(folderName=None) :
#	logger = logging.getLogger()
#	logger.setLevel(logging.DEBUG)
#	formatter = logging.Formatter('%(message)s') 
#
#	# the 'RotatingFileHandler' object implements a log file that is automatically limited in size
#	if folderName != None :
#		fh = RotatingFileHandler( os.path.join(folderName, "log.log"), mode='a', maxBytes=100*1024*1024, backupCount=2, encoding=None, delay=0 )
#		fh.setLevel(logging.DEBUG)
#		fh.setFormatter(formatter)
#		logger.addHandler(fh)
#
#	ch = logging.StreamHandler()
#	ch.setLevel(logging.INFO)
#	ch.setFormatter(formatter)
#	logger.addHandler(ch)
#	
#	return

def main() :

	#dbname = 'wine'
	#dbname = 'diabetes'
	#dbname = 'Australian'
	#dbname = 'vehicle'
	#dbname = 'madelon'
	dbname = 'gisette'
	root = '../results/comparison/' + dbname + '/'
	classifier_name = 'LogisticRegression'
	classifier_class = LogisticRegression()
	#classifier_name = 'RandomForestClassifier'
	#classifier_class = RandomForestClassifier()
	n_splits = 10
	kf = 1
	seed = 42
	figsize = [6, 4]
	
	## create folder with unique name
	#folderName = root + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")  
	#folderName += "-statistical-analysis"
	#if not os.path.exists(folderName) : os.makedirs(folderName)
	## initialize logging, using a logger that smartly manages disk occupation
	#initialize_logging(folderName)
	
	
	for key, db in openml.datasets.list_datasets().items():
		if db['name'] == dbname:
			
			# load different datasets, prepare them for use
			if dbname == 'madelon':
				X, y = datasets.make_classification(n_samples=4400,
													n_features=500,
													n_informative=5,
													n_redundant=15,
													shuffle=False,
													random_state=seed)
			else:
				db = openml.datasets.get_dataset(db['did'])
				X, y, attribute_names = db.get_data(
						   target=db.default_target_attribute,
						      return_attribute_names=True)
				try:
					X = X.toarray()
				except:
					print("X was already a numpy array...")
			
			
			break
	
	solution_sets = pd.DataFrame()
		
	for root, dirs, files in os.walk(root):
		for file in files:
			if len(file.split('_')) == 5 and file.startswith('final'):
			
				db = file.split('_')[2]
				m = file.split('_')[3]
				c = file.split('_')[4].split('.')[0]
				if db==dbname and c==classifier_name:
					
					sol = read_csv(root+file)
					sol['method'] = m
					solution_sets = pd.concat( [solution_sets, sol] )
	
	solution_sets = solution_sets.reset_index(drop=True)
	solution_sets = solution_sets[solution_sets['size']<3127]	
	
	split_index = 1
	for k in range(0, kf):
		skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=k)
		for train_index, test_index in skf.split(X, y) :
			
			X_train, y_train = X[train_index], y[train_index]
			X_test, y_test = X[test_index], y[test_index]
			print("\tSplit %d" %(split_index))
		
			# rescale data
			scaler = StandardScaler()
			sc = scaler.fit(X_train)
			X_train = sc.transform(X_train)
			X_test = sc.transform(X_test)
			
			accuracy = []
			for index, row in solution_sets.iterrows():
				
				print("\tSplit %d - Sol %d of %d" %(split_index, index, len(solution_sets)))
				
				s = row[0]
				s = s.replace('(', '')
				s = s.replace(' )', '')
				s = s.replace(')', '')
				feature_set = [int(x) for x in s.split("  ")]
				
				X_train_r = X_train[:, feature_set]
				X_test_r = X_test[:, feature_set]
				
				# compute classification accuracy
				referenceClassifier = copy.deepcopy(classifier_class)
				referenceClassifier.fit(X_train_r, y_train)
				accuracy.append( referenceClassifier.score( X_test_r, y_test ) )
			
			solution_sets['fold_'+str(split_index)] = accuracy
			split_index = split_index + 1
		
	size_unique = solution_sets['size'].unique()
	method_unique = solution_sets['method'].unique().tolist()
	#unique_feature_sets = solution_sets.drop_duplicates('feature-set')
	#for index, row in unique_feature_sets.iterrows():
	#	unique_feature_sets.at[index,'feature-set'] = [ int(i) for i in row[0] if (i!='(' and i!=')' and i!=' ') ]
	
	solution_sets['avg-accuracy'] = solution_sets.iloc[:, 3:].mean(axis=1)
	count_table = pd.DataFrame(columns=method_unique)
	
	
	
	################## overall count
	
	best_solution = solution_sets.loc[ solution_sets['avg-accuracy'].idxmax() ]
	best_sol = best_solution.iloc[3:-1].values
	best_sol_size = best_solution['size']
	solution_sets = solution_sets.sort_values(['avg-accuracy'], ascending=False)
	solution_sets_filtered = solution_sets[solution_sets['size']<=best_sol_size]
	
	l = len(solution_sets_filtered)
	M_pvalue = np.zeros((l,1))
	P_value = np.zeros((l,1))
	block = []
	
	for i in range(0, l):
		x = solution_sets_filtered.iloc[i, 3:-1].values
		
		t, p = stats.ttest_ind(best_sol, x, equal_var=False)
		val = p<0.05
		
		M_pvalue[i] = val
		P_value[i] = p
	
	solution_sets_filtered['block'] = M_pvalue
	
	null_df = pd.DataFrame()
	null_df['count'] = [0,0,0]
	null_df['method'] = method_unique
	
	solution_sets_b = solution_sets_filtered[solution_sets_filtered['block']==0]
	group_count = solution_sets_b.groupby(['method'])['method'].agg(['count'])
	group_count = group_count.reset_index()
	group_count = pd.concat( [null_df, group_count] )
	group_count = group_count.drop_duplicates(subset=['method'], keep='last')
	group_count = group_count.sort_values(['method'])
	
	plt.figure(figsize=figsize)
	ax = plt.gca()
	group_count.plot(kind='bar', x='method', y='count', ax=ax, alpha=0.5, legend=False)
	plt.ylabel('#best')
	plt.title("%s, best solutions" %(dbname))
	plt.tight_layout()
	plt.savefig(root + dbname + "_" + classifier_name + "_as_best.png")
	plt.savefig(root + dbname + "_" + classifier_name + "_as_best.pdf")
	plt.show()
	
	## exhaustive scatterplot
	#dataset = root + dbname + '_exhaustive_logisticRegression_randomForest_100_folds.csv'
	#try:
	#	df = pd.read_csv(dataset)
	#	filter_col = [col for col in df if col.startswith('RandomForest')]
	#	df = df.drop(filter_col, axis=1)
	#	
	#	sol_x = []
	#	sol_y = []
	#	
	#	best_solution = df.loc[ df[classifier_name + '_avg_test_accuracy'].idxmax() ]
	#	best_sol = best_solution.iloc[4:].values
	#	solution_sets_s = df.sort_values([classifier_name + '_avg_test_accuracy'], ascending=False)
	#	
	#	l = len(solution_sets_s)
	#	M_pvalue = np.zeros((l,1))
	#	P_value = np.zeros((l,1))
	#	block = []
	#	
	#	for j in range(0, l):
	#		#x1 = solution_sets_s.iloc[i, 4:].values
	#		x2 = solution_sets_s.iloc[j, 4:].values
	#		
	#		t, p = stats.ttest_ind(best_sol, x2, equal_var=False)
	#		val = p < 5e-2
	#		
	#		M_pvalue[j] = val
	#		P_value[j] = p
	#	
	#	solution_sets_s['block'] = M_pvalue
	#	solution_sets_b = solution_sets_s[solution_sets_s['block']==0]
	#	
	#	for index, row in solution_sets_b.iterrows() :
	#		sol_x.append( int(row['size']) )
	#		sol_y.append( 1.0 - float(row[classifier_name + '_avg_test_accuracy']) )
	#	
	#	x = []
	#	y = []
	#	for index, row in df.iterrows() :
	#		x.append( int(row['size']) )
	#		y.append( 1.0 - float(row[classifier_name + '_avg_test_accuracy']) )
	#	
	#	fig = plt.figure(figsize=figsize)
	#	ax = fig.add_subplot(111)
	#	ax.scatter(x, y, marker='.', alpha=0.4, color='gray', label="exhaustive")
	#	ax.scatter(sol_x, sol_y, marker='.', alpha=0.8, color='r', label="best solution sets", zorder=10)
	#	ax.scatter(best_solution['size'], 1-best_solution[classifier_name + '_avg_test_accuracy'], label='best', marker='o', color='g', s=50, zorder=10)
	#	plt.legend(loc='upper right', framealpha=1)
	#	plt.title("%s, exhaustive search" %(dbname))
	#	plt.xlabel('#features')
	#	plt.ylabel('avg test error')
	#	plt.tight_layout()
	#	plt.savefig(root + dbname + "_" + classifier_name + "_exhaustive.png")
	#	plt.savefig(root + dbname + "_" + classifier_name + "_exhaustive.pdf")
	#	plt.show()
	#	
	#	################## method by method
	#		
	#	
	#	q = 0
	#	group_count = pd.DataFrame()
	#	cols = ['feature_set', 'size_x', 'method', classifier_name + '_avg_test_accuracy']
	#	
	#	for mu in method_unique:
	#		
	#		try:
	#			fig = plt.figure(figsize=figsize)
	#			ax = fig.add_subplot(111)
	#			
	#			dm = solution_sets.loc[ solution_sets['method']==mu ]
	#			dm = dm.rename(columns={"feature-set": "feature_set"})
	#			dm = pd.merge(dm, solution_sets_b, how='inner', on=['feature_set'])
	#			dm = dm.sort_values(by=['size_x'])
	#			dm['y'] = 1-dm[classifier_name + '_avg_test_accuracy']
	#			dm.plot.scatter(x='size_x', y='y', ax=ax, label=mu, marker='*', color='r', s=200, zorder=5)
	#			ax.scatter(best_solution['size'], 1-best_solution[classifier_name + '_avg_test_accuracy'], label='best', marker='o', color='g', s=50, zorder=10)
	#			ax.scatter(x, y, marker='.', alpha=0.2, color='gray', label="exhaustive")
	#			plt.legend(loc='upper right', framealpha=1)
	#			plt.title("%s, %s, %d solutions" %(dbname, mu, len(dm)))
	#			plt.xlabel('#features')
	#			plt.ylabel('avg test error')
	#			plt.tight_layout()
	#			plt.savefig(root + dbname + "_" + classifier_name + "_" + mu + "_exhaustive.png")
	#			plt.savefig(root + dbname + "_" + classifier_name + "_" + mu + "_exhaustive.pdf")
	#			plt.show()
	#			
	#			group_count = group_count.append(dm[ cols ])
	#			
	#		except:
	#			print("No best sol")
	#	
	#		q = q+1
	#	
	#	
	#	################## overall count
	#	
	#	group_count = group_count[ group_count['size_x'] <= best_solution['size'] ]
	#	group_count = group_count.groupby(['method'])['method'].agg(['count'])
	#	group_count = group_count.reset_index()
	#	group_count = pd.concat( [null_df, group_count] )
	#	group_count = group_count.drop_duplicates(subset=['method'], keep='last')
	#	group_count = group_count.sort_values(['method'])
	#	
	#	plt.figure(figsize=figsize)
	#	ax = plt.gca()
	#	group_count.plot(kind='bar', x='method', y='count', ax=ax, alpha=0.5, legend=False)
	#	plt.ylabel('#best')
	#	plt.title("%s, best solutions" %(dbname))
	#	plt.tight_layout()
	#	plt.savefig(root + dbname + "_" + classifier_name + "_as_best.png")
	#	plt.savefig(root + dbname + "_" + classifier_name + "_as_best.pdf")
	#	plt.show()
	#
	#	
	#except:
	#	print('error')
		
	
	
	############### for madelon only!!!
		
	if dbname == 'madelon':
		
		good_fset = np.arange(0, 20)
		madelon_best = pd.DataFrame(columns=method_unique)
		for i in range(0, 20):
			madelon_best.loc[len(madelon_best)] = [0,0,0,0]
				
		bad_fset = np.arange(20, 500)
		madelon_worst = null_df.copy()
		
		for mu in method_unique:
			
			try:
				dm = solution_sets.loc[ solution_sets['method']==mu ]
				
				count = 0
				for index, row in dm.iterrows():
					s = row[0]
					s = s.replace('(', '')
					s = s.replace(' )', '')
					s = s.replace(')', '')
					feature_set = [int(x) for x in s.split("  ")]
					
					# good feature sets
					intersection_list_good = list(set(good_fset) & set(feature_set))
					for n in intersection_list_good:
						madelon_best.loc[madelon_best.index == n, [mu]] += 1
						
					# bad feature sets
					intersection_list_bad = list(set(bad_fset) & set(feature_set))
					madelon_worst.loc[madelon_worst['method'] == mu, ['count']] += len(intersection_list_bad)
					
					count += len(feature_set)
					
				madelon_best[mu] = madelon_best[mu] / len(dm)
				madelon_worst.loc[madelon_worst['method'] == mu, ['count']] /= count
				
				plt.figure(figsize=figsize)
				ax = plt.gca()
				madelon_best.plot(kind='bar', y=mu, alpha=0.5, ax=ax, legend=False, color='g')
				plt.ylabel('selection frequency')
				plt.title("%s, %s, genuine features" %(dbname, mu))
				plt.tight_layout()
				plt.savefig(root + dbname + "_" + classifier_name + "_" + mu + "_true_sets.png")
				plt.savefig(root + dbname + "_" + classifier_name + "_" + mu + "_true_sets.pdf")
				plt.show()
				
			except:
				print("error")
				
				
		madelon_worst = madelon_worst.sort_values(['method'])
		
		plt.figure(figsize=figsize)
		ax = plt.gca()
		madelon_worst.plot(kind='bar', x='method', y='count', ax=ax, alpha=0.5, legend=False)
		plt.ylabel('selection frequency')
		plt.title("%s, probes" %(dbname))
		plt.tight_layout()
		plt.savefig(root + dbname + "_" + classifier_name + "_probes.png")
		plt.savefig(root + dbname + "_" + classifier_name + "_probes.pdf")
		plt.show()
	
	
	
	
	
	

if __name__ == "__main__" :
	sys.exit( main() )
	
	
	
	
	
	
	
	
#sem = []	
#for index, row in solution_sets.iterrows():
#	sem.append( np.std( row[3:-1] ) / np.sqrt( len(row[3:-1]) ) )
#avg_sem = np.mean(sem)
	
	
	
	
	
	
	
	
	
#	bad_fset = np.arange(20, 500)
#	
#	madelon_best = null_df.copy()
#	
#	for mu in method_unique:
#		
#		dm = solution_sets.loc[ solution_sets['method']==mu ]
#	
#		count = len(dm)
#		for index, row in dm.iterrows():
#			s = row[0]
#			s = s.replace('(', '')
#			s = s.replace(' )', '')
#			s = s.replace(')', '')
#			feature_set = [int(x) for x in s.split("  ")]
#			count_list = list(set(bad_fset) & set(feature_set))
#			
#			if len(count_list) > 0:
#				count = count - 1
#		
#		madelon_best.loc[madelon_best['method'] == mu, ['count']] = count
#	
#	madelon_best = madelon_best.sort_values(['method'])
#	
#	plt.figure(figsize=figsize)
#	ax = plt.gca()
#	madelon_best.plot(kind='bar', x='method', y='count', ax=ax, alpha=0.5, legend=False)
#	plt.ylabel('#best')
#	plt.title("%s, genuine sets" %(dbname))
#	plt.tight_layout()
#	plt.savefig(root + dbname + "_" + classifier_name + "_true_best.png")
#	plt.savefig(root + dbname + "_" + classifier_name + "_true_best.pdf")
#	plt.show()










#sol_x = []
#sol_y = []
#
#for s in size_unique:
##	s=3
#	print("\n\n*** size %d ***\n" %(s))
#	
##	s = 1
#	solution_sets_s = solution_sets[solution_sets['size']==s]
#	best_sol = solution_sets_s.loc[ solution_sets_s['avg-accuracy'].idxmax() ]
#	best_sol = best_sol.iloc[3:-1].values
#	solution_sets_s = solution_sets_s.sort_values(['avg-accuracy'], ascending=False)
#	
#	l = len(solution_sets_s)
#	M_pvalue = np.zeros((l,l))
#	P_value = np.zeros((l,l))
#	block = []
#	
##	for i in range(0, l):
##		for j in range(i+1, l):
##			x1 = solution_sets_s.iloc[i, 3:-1].values
##			x2 = solution_sets_s.iloc[j, 3:-1].values
##			
##			t, p = stats.ttest_ind(x1, x2)
##			val = p<0.05
##			
##			M_pvalue[i,j] = val
##			P_value[i,j] = p
##	
##	k = 1
##	block = np.zeros(l)
##	i = 0
##	while i < l:
##		j = np.argmax(M_pvalue[i,:])
##		if j == 0:
##			block[i:] = k
##			break
##		block[i:j] = k
##		k = k+1
##		i = j
#	i = 0
#	for j in range(0, l):
##		x1 = solution_sets_s.iloc[i, 3:-1].values
#		x2 = solution_sets_s.iloc[j, 3:-1].values
#		
#		t, p = stats.ttest_ind(best_sol, x2, equal_var=False)
#		val = p<0.05
#		
#		M_pvalue[j] = val
#		P_value[j] = p
#	
#	solution_sets_s['block'] = M_pvalue
#	
#	solution_sets_b = solution_sets_s[solution_sets_s['block']==0]
#	
#	for index, row in solution_sets_b.iterrows() :
#		sol_x.append( int(row['size']) )
#		sol_y.append( 1.0 - float(row['avg-accuracy']) )
#	
#	group_count = solution_sets_b.groupby(['method'])['method'].agg(['count']).transpose()
#	group_count = group_count.set_index(pd.Index([s]))
#	
#	count_table = pd.concat( [count_table, group_count] )
#	
#count_table = count_table.fillna(0)
#count_table['size'] = np.arange(1, len(count_table)+1)
#
##plt.figure(figsize=figsize)
##ax = plt.gca()
##count_table.plot(kind='line', x='size', y='EvoFS', marker='o', color='red', ax=ax, alpha=0.3)
##count_table.plot(kind='line', x='size', y='RFE', marker='o', color='black', ax=ax, alpha=0.3)
##count_table.plot(kind='line', x='size', y='anova', marker='o', color='green', ax=ax, alpha=0.3)
##count_table.plot(kind='line', x='size', y='mi', marker='o', color='blue', ax=ax, alpha=0.3)
##plt.ylabel('#solutions in best group')
##plt.legend()
##plt.title("%s, %s, %d-fold CV" %(dbname, classifier_name, n_splits))
##plt.tight_layout()
##plt.savefig(root + dbname + "_" + classifier_name + ".png")
##plt.savefig(root + dbname + "_" + classifier_name + ".pdf")
##plt.show()






















#	worst_among_best = solution_sets_b.loc[ solution_sets_b['avg-accuracy'].idxmin() ]
#	x = [0, best_sol_size+0.5]
#	y = [ 1-worst_among_best['avg-accuracy'] ] * 2
#
#	c = ['m', 'g', 'b', 'k']
#	m = ['<', '>', '^', 'v']
#	q = 0
#	
#	fig = plt.figure(figsize=figsize)
#	ax = fig.add_subplot(111)
#	for mu in method_unique:
#		dm = solution_sets.loc[ solution_sets['method']==mu ]
#		dm = dm.sort_values(by=['size'])
#		if mu == 'EvoFS':
#			dm = dm.groupby(['size'])['avg-accuracy'].agg(['max'])
#			plt.plot(dm.index.values, 1-dm['max'].values, color=c[q], marker=m[q], label=mu, alpha=.8)
#		else:
#			plt.plot(dm['size'].values, 1-dm['avg-accuracy'].values, color=c[q], marker=m[q], label=mu, alpha=.8)
#		q = q+1
#	plt.plot(x, y, 'r--', label='best solutions\' line')
#	plt.legend()
#	plt.title("%s, best solutions" %(dbname))
#	plt.xlabel('#features')
#	plt.ylabel('avg test error')
#	plt.xlim([0.5, best_sol_size + 0.5])
#	plt.tight_layout()
#	plt.savefig(root + dbname + "_" + classifier_name + "_scatter.png")
#	plt.savefig(root + dbname + "_" + classifier_name + "_scatter.pdf")
#	plt.show()

#count_table['size'] = np.arange(1, len(count_table)+1)
#
#plt.figure(figsize=figsize)
#sns.countplot(x="size",  data=count_table)
		
#for j in range(0, l):
#	solution_sets['pval_sol_'+str(j)] = M_pvalue[:, j]

					
#		c = ['r', 'g', 'b', 'k']
#		m = ['<', '>', '^', 'v']
#		
#		fig = plt.figure(figsize=figsize)
#		ax = fig.add_subplot(111)
#		
#		i = 0
#		for key, value in solution_sets.items():
#			ax.scatter(value[:, 0], value[:, 1], color=c[i], marker=m[i], alpha=0.6, label="Best feature sets for " + key)
#			i = i + 1
#		
#		ax.set_xlabel("number of features")
#		ax.set_ylabel("average test error")
#		ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
#		ax.set_title("%s, %s, fold %d" %(dbname, classifier, f) )
#	#	ax.set_xlim([0.5, 8.5])
##		ax.set_xlim([0.5, 100])
#		
#		plt.tight_layout()
#		plt.savefig(root + dbname + "_" + str(f) + "_" + classifier + ".png")
#		plt.savefig(root + dbname + "_" + str(f) + "_" + classifier + ".pdf")
#		plt.show()
#		plt.close(fig)

#	return
#
