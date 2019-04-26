import warnings
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from voting_booth import voting_booth
import numpy as np
import json
import copy
from datetime import datetime

from new_mgs import MGS
from board_opinion import board_opinion as bo
from filters import perf_filter


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

def run_board_ensemble(X,y):
	start = datetime.now()
	#Data
	X_train, X_validation, y_train, y_validation = train_test_split(X,y)

	#Models
	# run block of code and catch warnings
	with warnings.catch_warnings():
		# ignore all caught warnings
		warnings.filterwarnings("ignore")
		board = bo(n_jobs = -1, time_serie = False, nbr_train_test_split = 3, scoring = "accuracy")
		board = board.fit(X_train,y_train)
		train_preds = board.predict_probas(X_train)
		preds = board.predict_probas(X_validation)

	#print("clf done",preds.shape[0])
	clfs_acc = np.array([accuracy_score(y_validation, np.argmax(pred,axis = 1)) for pred in preds])

	clf_time = datetime.now()
	#print("CLF time",clf_time - start, preds.shape[0])

	#Filter
	filt = perf_filter(accuracy_score)
	filt = filt.selection(y_train, train_preds)
	train_preds = filt.filter(train_preds, nbr_to_filter = 12)
	preds = filt.filter(preds, nbr_to_filter = 12)


	ensemble_acc_dic = {}
	#print("MGR")
	mgs = MGS(score_function = accuracy_score, n_jobs = -1)
	mgs = mgs.fit(train_preds, y_train)
	pred = mgs.predict_proba(preds)
	ensemble_acc_dic["MGS"] = accuracy_score(y_validation, np.argmax(pred,axis = 1))
	
	mgs_time = datetime.now()
	#print("mgs",mgs_time - clf_time)

	#MAJ VOTING
	#print("Maj_voting")
	pred = voting_booth().vote(copy.deepcopy(preds))
	ensemble_acc_dic["Maj_Voting"] = accuracy_score(y_validation, pred)

	maj_vote_time = datetime.now()
	#print("vote", maj_vote_time - mgs_time)

	train_preds = np.array([x.T for x in train_preds])
	preds = np.array([x.T for x in preds])
	train_preds = train_preds.reshape(-1,train_preds.shape[-1])
	preds = preds.reshape(-1,preds.shape[-1])
	#print("Random Forest")
	rf = RandomForestClassifier(n_estimators=100,n_jobs=-1)
	rf = rf.fit(train_preds.T,y_train)
	pred = rf.predict(preds.T)
	ensemble_acc_dic["RF"] = accuracy_score(y_validation, pred)

	rf_time = datetime.now()
	#print("rf",rf_time - maj_vote_time)

	return clfs_acc, ensemble_acc_dic

def main():
	nbr_iterations = 10
	clf_min, clf_max, clf_median, clf_avg = [],[],[],[]
	rf, mgs, maj_vote = [], [], []
	iris = datasets.load_iris()
	X = iris.data
	y = iris.target
	for n in range(nbr_iterations):
		clfs, ensembles_dico = run_board_ensemble(X,y)
		clf_min.append(clfs.min())
		clf_max.append(clfs.max())
		clf_median.append(np.median(clfs))
		clf_avg.append(np.average(clfs))
		rf.append(ensembles_dico["RF"])
		mgs.append(ensembles_dico["MGS"])
		maj_vote.append(ensembles_dico["Maj_Voting"])

		printProgressBar(n,nbr_iterations)

	with open("stats_results.txt", 'a') as f:
	    f.write('min: ' + str(np.average(clf_min)) + '\n')
	    f.write('max: ' + str(np.average(clf_max)) + '\n')
	    f.write('avg: ' + str(np.average(clf_avg)) + '\n')
	    f.write('med: ' + str(np.average(clf_median)) + '\n')
	    f.write("MGS: " + str(np.average(mgs)) + '\n')
	    f.write("Majority Voting: " + str(np.average(maj_vote)) + '\n')
	    f.write('Random Forest: ' + str(np.average(rf)) + '\n')
	    f.write('\n')


main()
