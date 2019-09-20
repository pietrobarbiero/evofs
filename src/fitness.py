# Script with all the fitness functions that can be used for our beloved EA.
# by Pietro Barbiero and Alberto Tonda, 2019 <alberto.tonda@gmail.com>

import copy

from sklearn.metrics import accuracy_score

# performs a cross validation, given folds and a classifier
def cross_validation(folds, X, y, reference_classifier, return_error=False) :
	
	test_accuracies = []
	classifier = copy.deepcopy(reference_classifier)

	for train_index, test_index in folds :
		X_train, y_train = X[train_index], y[train_index]
		X_test, y_test = X[test_index], y[test_index]
		
		classifier.fit(X_train, y_train)
		y_test_pred = classifier.predict(X_test)
		test_accuracies.append( accuracy_score(y_test, y_test_pred) )
	
	# if the flag is set, return (1.0 - accuracy) for each fold (basically, the error)
	if return_error : test_accuracies = [ (1.0 - t) for t in test_accuracies ]
	
	return test_accuracies
