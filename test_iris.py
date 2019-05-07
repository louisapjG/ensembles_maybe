import warnings
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from voting_booth import voting_booth
import numpy as np
import json
import copy
from datetime import datetime

from bokeh.io import show, output_file
from bokeh.plotting import figure

from new_mgs import MGS
from board_opinion import board_opinion as bo
from filters import perf_filter, cds_filter, ruler_filter

"""
TODO 
	0. create cd/cds class
	1. Create graphs for results visualization??? 1 graph per dataset? Has a spread for each type clf, and each ensemble. Bar graph for datasets
	2. Add Filtering: CDS
	3. Filtering read + apply other methods
	?-1. Transform run_board_ensemble in a class
TODO GIT:
	1. Assess filtering methods against best initial classifier and unfiltered majority voting.
	2. Graph of performance change when increassing the number of clf filtered
"""


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 50, fill = '█'):
	iteration+=1
	"""
	Call in a loop to create terminal progress bar
	@params:
	    iteration   - Required  : current iteration (Int)
	    total       - Required  : total iterations (Int)
	    prefix      - Optional  : prefix string (Str)
	    suffix      - Optional  : suffix string (Str)
	    decimals    - Optional  : positive number of decimals in percent complete (Int)
	    length      - Optional  : character length of bar (Int)
	    fill        - Optional  : bar fill character (Str)
	"""
	percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
	filledLength = int(length * iteration // total)
	bar = fill * filledLength + '-' * (length - filledLength)
	print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
	# Print New Line on Complete
	if iteration == total: 
		print()

def run_board_ensemble(X,y, time_serie = False, n_splits = 5):
	start = datetime.now()
	
	#Data
	#X_train, X_validation, y_train, y_validation = train_test_split(X,y)
	if(time_serie):
		splits = TimeSeriesSplit(n_splits = n_splits)
	else:
		splits = StratifiedKFold(n_splits = n_splits)


	ensemble_acc_dic = {}

	clfs_preds = []
	best_clf = []

	# mgs_preds = []
	# maj_vote_preds = []
	# rf_preds = []
	#
	mgs_preds = {'cds':[], 'ruler':[], 'perf':[]}
	maj_vote_preds = {'cds':[], 'ruler':[], 'perf':[]}
	rf_preds = {'cds':[], 'ruler':[], 'perf':[]}
	
	clfs_acc = []

	y_of_preds = []
	for train, test in splits.split(X,y):
		X_train, y_train = X[train], y[train]
		X_validation, y_validation = X[test], y[test]

		y_of_preds = y_of_preds + list(y_validation)

		#Models
		# run block of code and catch warnings
		with warnings.catch_warnings():
			# ignore all caught warnings
			warnings.filterwarnings("ignore")
			board = bo(n_jobs = -1, time_serie = False, nbr_train_test_split = 3, scoring = "accuracy")
			train_preds,y_trained = board.fit(X_train,y_train, predict_training_probas = True)
			preds = board.predict_probas(X_validation)
			if(clfs_preds == []):
				clfs_preds = np.argmax(preds,axis = -1).tolist()
			else:
				for ind,pred in enumerate(preds):
					clfs_preds[ind].extend(np.argmax(pred,axis = -1).tolist())
		
		clf_time = datetime.now()
		#print("CLF time",clf_time - start, preds.shape[0])
		filters_dico = {}
		#Testing cds filter
		filt = cds_filter(accuracy_score)
		filt = filt.selection(y_trained, train_preds)
		filters_dico["cds"] = filt

		#Testing cds/perf ruling filter
		filt = ruler_filter(accuracy_score)
		filt = filt.selection(y_trained, train_preds)
		filters_dico["ruler"] = filt

		#Filter
		filt = perf_filter(accuracy_score)
		filt = filt.selection(y_trained, train_preds)
		filters_dico["perf"] = filt
		#Select the best clf at training and record it's testing scores
		best_pred = filt.filter(preds, nbr_to_filter = 1)
		best_clf.extend(np.argmax(best_pred[0], axis = 1))

		#For each filter
		for filter_name in filters_dico:
			#Do
			train_preds_filtered = filt.filter(train_preds, nbr_to_filter = 12)
			preds_filtered = filt.filter(preds, nbr_to_filter = 12)
			
			#MGS
			#print("MGS")
			mgs = MGS(score_function = accuracy_score, n_jobs = -1)
			mgs = mgs.fit(train_preds_filtered, y_trained)
			pred = mgs.predict_proba(preds_filtered)
			#mgs_preds.extend(np.argmax(pred, axis = 1).tolist())
			mgs_preds[filter_name].extend(np.argmax(pred, axis = 1).tolist())
			
			mgs_time = datetime.now()
			#print("mgs",mgs_time - clf_time)

			#MAJ VOTING
			#print("Maj_voting")
			pred = voting_booth().vote(copy.deepcopy(preds_filtered))
			#maj_vote_preds.extend(pred.tolist())
			maj_vote_preds[filter_name].extend(pred.tolist())

			maj_vote_time = datetime.now()
			#print("vote", maj_vote_time - mgs_time)

			#RANDOM FOREST
			train_preds_filtered = np.array([x.T for x in train_preds_filtered])
			preds_filtered = np.array([x.T for x in preds_filtered])
			train_preds_filtered = train_preds_filtered.reshape(-1,train_preds_filtered.shape[-1])
			preds_filtered = preds_filtered.reshape(-1,preds_filtered.shape[-1])
			#print("Random Forest")
			rf = RandomForestClassifier(n_estimators=100,n_jobs=-1)
			rf = rf.fit(train_preds_filtered.T,y_trained)
			pred = rf.predict(preds_filtered.T)
			#rf_preds.extend(pred.tolist())
			rf_preds[filter_name].extend(pred.tolist())
		#rf_time = datetime.now()

	ensemble_acc_dic["Best_clf"] = accuracy_score(y_of_preds,best_clf)
	for filter_name in mgs_preds:
		ensemble_acc_dic["MGS_"+filter_name] = accuracy_score(y_of_preds, mgs_preds[filter_name])
		ensemble_acc_dic["Maj_Voting_"+filter_name] = accuracy_score(y_of_preds, maj_vote_preds[filter_name])
		ensemble_acc_dic["RF_"+filter_name] = accuracy_score(y_of_preds, rf_preds[filter_name])

	clfs_acc = np.array([accuracy_score(y_of_preds, pred) for pred in clfs_preds])

	return clfs_acc, ensemble_acc_dic

def main():
	nbr_iterations = 3
	#rf, mgs, maj_vote, best_clf = [], [], [], []
	ensembles_results = {}
	iris = datasets.load_iris()#load_digits()#
	output_file_name = "stats_results_iris.txt" # "stats_results_digits.txt"#
	X = iris.data
	y = iris.target
	
	for n in range(nbr_iterations):
		clfs, ensembles_dico = run_board_ensemble(X,y, time_serie = False, n_splits = 3)
		#best_clf.append(ensembles_dico["best_clf"])
		for key in ensembles_dico:
			if key in ensembles_results:
				ensembles_results[key].extend([ensembles_dico[key]])
			else:
				ensembles_results[key] = [ensembles_dico[key]]

		# rf.append(ensembles_dico["RF"])
		# mgs.append(ensembles_dico["MGS"])
		# maj_vote.append(ensembles_dico["Maj_Voting"])

		printProgressBar(n,nbr_iterations)

	with open(output_file_name, 'a') as f:
		#f.write("Best_clf: " + str(np.average(best_clf)) + '\n')
		for key in ensembles_results:
			f.write(key + ": " + str(np.average(ensembles_results[key])) + '\n')
		    # f.write("Majority Voting: " + str(np.average(maj_vote)) + '\n')
		    # f.write('Random Forest: ' + str(np.average(rf)) + '\n')
		f.write('\n')

	#Graph bokeh
	output_file("bar_sorted.html")
	values = [np.average(ensembles_results[key]) for key in ensembles_results]
	col_names = [key  for key in ensembles_results]
	sorted_names = sorted(col_names, key=lambda x: values[col_names.index(x)])
	p = figure(x_range=sorted_names, plot_height=350, title="Ensembles performances", toolbar_location=None, tools="hover", tooltips="$name @col_names: @$name")
	p.vbar(x=col_names, top=values, width=0.9)
	p.xgrid.grid_line_color = None
	p.y_range.start = 0
	show(p)



main()

#
