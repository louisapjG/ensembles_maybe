import warnings
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from board_opinion import board_opinion as bo
import numpy as np
from new_mgs import MGS
import json

def run_board_ensemble(X,y):
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

	clfs_acc = np.array([accuracy_score(y_validation, np.argmax(pred,axis = 1)) for pred in preds])

	ensemble_acc_dic = {}
	# print("MGR")
	mgs = MGS(score_function = accuracy_score, n_jobs = -1)
	mgs = mgs.fit(train_preds, y_train)
	pred = mgs.predict_proba(preds)
	ensemble_acc_dic["MGS"] = accuracy_score(y_validation, np.argmax(pred,axis = 1))

	
	train_preds = np.array([x.T for x in train_preds])
	preds = np.array([x.T for x in preds])
	train_preds = train_preds.reshape(-1,train_preds.shape[-1])
	preds = preds.reshape(-1,preds.shape[-1])
	#print("Random Forest")
	rf = RandomForestClassifier(n_estimators=100,n_jobs=-1)
	rf = rf.fit(train_preds.T,y_train)
	pred = rf.predict(preds.T)
	ensemble_acc_dic["RF"] = accuracy_score(y_validation, pred)

	return clfs_acc, ensemble_acc_dic



def main():
	clf_min, clf_max, clf_median, clf_avg = [],[],[],[]
	rf, mgs = [],[]
	iris = datasets.load_iris()
	X = iris.data
	y = iris.target
	for n in range(10):
		clfs, ensembles_dico = run_board_ensemble(X,y)
		clf_min.append(clfs.min())
		clf_max.append(clfs.max())
		clf_median.append(np.median(clfs))
		clf_avg.append(np.average(clfs))
		rf.append(ensembles_dico["RF"])
		mgs.append(ensembles_dico["MGS"])

	with open("stats_results.txt", 'a') as f:
	    f.write('min: ' + str(np.average(clf_min)) + '\n')
	    f.write('max: ' + str(np.average(clf_max)) + '\n')
	    f.write('avg: ' + str(np.average(clf_avg)) + '\n')
	    f.write('med: ' + str(np.average(clf_median)) + '\n')
	    f.write('Random Forest: ' + str(np.average(rf)) + '\n')
	    f.write("MGS: " + str(np.average(mgs)) + '\n')


main()
