#https://app.scholarjet.com/challenges/Xxj
import warnings
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, make_scorer
from sklearn import svm, ensemble, linear_model, naive_bayes, discriminant_analysis, gaussian_process, neighbors, tree
from sklearn.cluster import KMeans, FeatureAgglomeration
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler

from voting_booth import voting_booth
import numpy as np
import json
import copy
from datetime import datetime

from bokeh.io import show
from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import LabelSet, ranges
from bokeh.palettes import PuBu

from new_mgs import MGS
from board_opinion import board_opinion as bo
from filters import perf_filter, cds_filter, ruler_filter

import pandas as pd

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

def run_board_ensemble(X,y, dic_params_board, time_serie = False, n_splits = 5, nbr_train_test_split = 3, nbr_to_filter = 12, performance = accuracy_score):
	performance_sk = make_scorer(performance)
	start = datetime.now()

	dic_params_board["time_serie"] = time_serie
	dic_params_board["nbr_train_test_split"] = nbr_train_test_split
	dic_params_board["scoring"] = performance_sk

	
	#Data
	if(time_serie):
		splits = TimeSeriesSplit(n_splits = n_splits)
	else:
		splits = StratifiedKFold(n_splits = n_splits)


	ensemble_acc_dic = {}

	clfs_preds = []
	best_clf = []

	mgs_preds = {'cds':[], 'ruler':[], 'perf':[]}
	maj_vote_preds = {'cds':[], 'ruler':[], 'perf':[]}
	rf_preds = {'cds':[], 'ruler':[], 'perf':[]}
	clf2_preds = {'cds':[], 'ruler':[], 'perf':[]}
	
	clfs_acc = []

	y_of_preds = []
	#print(datetime.now())
	for train, test in splits.split(X,y):
		X_train, y_train = X[train], y[train]
		X_validation, y_validation = X[test], y[test]

		y_of_preds = y_of_preds + list(y_validation)

		#Models
		# run block of code and catch warnings
		with warnings.catch_warnings():
			# ignore all caught warnings
			warnings.filterwarnings("ignore")
			board = bo(**dic_params_board)
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
		filt = cds_filter(performance)
		filt = filt.selection(y_trained, train_preds)
		filters_dico["cds"] = filt
		cds_time = datetime.now()
		#print("cds time", cds_time - clf_time)

		#Testing cds/perf ruling filter
		filt = ruler_filter(performance)
		filt = filt.selection(y_trained, train_preds)
		filters_dico["ruler"] = filt
		ruler_time = datetime.now()
		#print("ruler time", ruler_time - cds_time)

		#Filter
		filt = perf_filter(performance)
		filt = filt.selection(y_trained, train_preds)
		filters_dico["perf"] = filt
		#Select the best clf at training and record it's testing scores
		best_pred = filt.filter(preds, nbr_to_filter = 1)
		best_clf.extend(np.argmax(best_pred[0], axis = 1))
		best_time = datetime.now()
		#print("best clf time", best_time - ruler_time)

		#For each filter
		for filter_name in filters_dico:
			#Do
			train_preds_filtered = filters_dico[filter_name].filter(train_preds, nbr_to_filter = nbr_to_filter)
			preds_filtered = filters_dico[filter_name].filter(preds, nbr_to_filter = nbr_to_filter)
			
			#MGS
			#print("MGS")
			# mgs = MGS(score_function = performance, n_jobs = -1)
			# mgs = mgs.fit(train_preds_filtered, y_trained)
			# pred = mgs.predict_proba(preds_filtered)
			# mgs_preds[filter_name].extend(np.argmax(pred, axis = 1).tolist())
			# mgs_time = datetime.now()
			#print("mgs",mgs_time - clf_time)

			#MAJ VOTING
			#print("Maj_voting")
			pred = voting_booth().vote(copy.deepcopy(preds_filtered))
			maj_vote_preds[filter_name].extend(pred.tolist())
			maj_vote_time = datetime.now()
			#print("vote", maj_vote_time - mgs_time)

			#Reshaping of train_preds_filtered and preds filtered from (a,b,c) to (a*c,b).
			#a: number classifiers
			#b: number of events
			#c: value for each target class
			train_preds_filtered = np.array([x.T for x in train_preds_filtered])
			preds_filtered = np.array([x.T for x in preds_filtered])
			train_preds_filtered = train_preds_filtered.reshape(-1,train_preds_filtered.shape[-1])
			preds_filtered = preds_filtered.reshape(-1,preds_filtered.shape[-1])

			#RANDOM FOREST
			#print("Random Forest")
			rf = RandomForestClassifier(n_estimators=100,n_jobs=-1)
			rf = rf.fit(train_preds_filtered.T,np.ravel(y_trained))
			pred = rf.predict(preds_filtered.T)
			rf_preds[filter_name].extend(pred.tolist())
			rf_time = datetime.now()

			#New set of classifiers:
			with warnings.catch_warnings():
				# ignore all caught warnings
				warnings.filterwarnings("ignore")
				board = bo(**dic_params_board)
				train_preds2,y_trained2 = board.fit(train_preds_filtered.T,y_trained, predict_training_probas = True)
				pred = board.predict_probas(preds_filtered.T)
			#Filter the training to get best one
			filt2 = perf_filter(performance)
			filt2 = filt2.selection(y_trained2, train_preds2)
			#Select the best clf at training
			pred = np.argmax(filt2.filter(pred, nbr_to_filter = 1)[0], axis = 1)
			clf2_preds[filter_name].extend(pred.tolist())

	ensemble_acc_dic["BestClf"] = performance(y_of_preds,best_clf)
	for filter_name in mgs_preds:
		#ensemble_acc_dic["MGS_"+filter_name] = performance(y_of_preds, mgs_preds[filter_name])
		ensemble_acc_dic["MajVoting_"+filter_name] = performance(y_of_preds, maj_vote_preds[filter_name])
		ensemble_acc_dic["RF_"+filter_name] = performance(y_of_preds, rf_preds[filter_name])
		ensemble_acc_dic["BestClf2_"+filter_name] =  performance(y_of_preds, clf2_preds[filter_name])

	clfs_acc = np.array([performance(y_of_preds, pred) for pred in clfs_preds])

	return clfs_acc, ensemble_acc_dic

def display_results(results):
	
	averaged_results = {key:np.round(np.average(results[key]), decimals=4) for key in results}

	output_file("wayfar_jet.html")
	x = [key for key in averaged_results]
	y = np.array([averaged_results[key] for key in averaged_results])
	args = np.argsort(y)
	x = np.array(x)[args]
	y = y[args]

	source = ColumnDataSource(dict(x = x, y = y))
	x_label = "Filter_Ensemble set"
	y_label = "Performance"
	title = "Performance bar plot"
	p = figure(plot_width=1100, plot_height=400, tools="save",
			x_axis_label = x_label,
			y_axis_label = y_label,
			title=title,
			x_minor_ticks=2,
			x_range = source.data["x"],
			y_range= ranges.Range1d(start=np.min(y)-.25,end=np.max(y)+0.05))

	labels = LabelSet(x='x', y='y', text='y', level='glyph',
					x_offset = -50, y_offset = 0, source=source, render_mode='canvas')

	p.vbar(source=source,x='x',top='y',bottom=0,width=0.3,color=PuBu[7][2])

	p.add_layout(labels)
	show(p)

def import_data(path):
	df = pd.read_csv(path)

	df = df.fillna(0)

	df_target = df[["cuid","convert_30","revenue_30"]]
	df = df.drop(["convert_30","revenue_30",'Unnamed: 0'], axis = 1)

	#Set all columns types object to categorical
	cat_columns = df.select_dtypes(['object']).columns
	#df[cat_columns] = df[cat_columns].apply(lambda x: x.astype('category').cat.codes.astype('category'))
	for name_col in cat_columns:
		df_dummy = pd.get_dummies(df[name_col])
		df = df.drop(name_col,axis=1)
		df = pd.concat([df, df_dummy],axis = 1)

	df = df.set_index("cuid")
	df_target = df_target.set_index("cuid")
	return df, df_target

def main():
	nbr_iterations = 5
	n_splits = 3
	nbr_train_test_split = 5
	nbr_to_filter = 12

	performance = accuracy_score
	output_file_name = "stats_results_wayfair.txt"#"stats_results_iris.txt" # 
	#Get data
	X, y_two = import_data("/Users/Louis/Desktop/code/ensembles/applications/df_training_scholarjet.csv")
	#ONLY DOES CATEGORICAL
	y = y_two.drop( "revenue_30", axis = 1)

	X, y = X.values, y.values

	dic_params_board = {
		#List of variability cleanups methods
		"variabilities" : [
							#(VarianceThreshold(),{'threshold':[.1]}),
							(VarianceThreshold(),{'threshold':[.0]}),
							#(None,None),
						],

		#List of Normalizations methods
		"normalizations" : [
								(MinMaxScaler(),{'feature_range':[(0, 1)]}),
								(None,None),
							],
		
		#List of Dimensions reduction methods
		"dim_red" : [
						(PCA(),{'n_components':[0.25,0.1]}),
						(FeatureAgglomeration(),{'n_clusters':[30,10]}),
						(KMeans(),{'n_clusters':[10,30]}),
						(None,None),
					],
		
		#List of clf
		"clf_params" : [(svm.SVC(probability=True),{'C': [1], 'gamma': [0.7], 'kernel': ['rbf']}),
						(svm.SVC(probability=True),{'C':[1],'kernel':['linear']}),
						(ensemble.RandomForestClassifier(),{'n_estimators':[300]}),
						(ensemble.AdaBoostClassifier(),{'base_estimator':[ensemble.RandomForestClassifier(n_estimators = 150),],'n_estimators':[100]}),
						(tree.DecisionTreeClassifier(),{'criterion':["gini"]}),	
						#(neighbors.KNeighborsClassifier(),{'n_neighbors':[7]}),
						#MCONSUMPTION ON LARGET SETS(gaussian_process.GaussianProcessClassifier(),[{'random_state':[3]}]),
						(ensemble.GradientBoostingClassifier(),{'loss':['deviance'],'n_estimators':[200],'max_depth':[3]}),#(ensemble.GradientBoostingClassifier(),{'loss':['deviance'],'n_estimators':[200,100,300],'max_depth':[2,4,6]}),
						(ensemble.ExtraTreesClassifier(),{'n_estimators':[200]}),#(ensemble.ExtraTreesClassifier(),{'n_estimators':[50,100,200,300]}),
						##(naive_bayes.GaussianNB(),[{'priors':[None]}]),
						#Linear classifier
						#Naive Bayes
						#Decision tree
						],
		#n_jobs
		"n_jobs" : -1
	}
	
	ensembles_results = {}
	for n in range(nbr_iterations):
		clfs, ensembles_dico = run_board_ensemble(X,y, time_serie = False, n_splits = n_splits, nbr_train_test_split = nbr_train_test_split, nbr_to_filter = nbr_to_filter, performance = performance, dic_params_board = dic_params_board)
		for key in ensembles_dico:
			if key in ensembles_results:
				ensembles_results[key].extend([ensembles_dico[key]])
			else:
				ensembles_results[key] = [ensembles_dico[key]]

		printProgressBar(n,nbr_iterations)

	with open(output_file_name, 'a') as f:
		for key in ensembles_results:
			f.write(key + ": " + str(np.average(ensembles_results[key])) + '\n')
		f.write('\n')

	
	averaged_results = {key:np.average(ensembles_results[key]) for key in ensembles_results}

	#Graph bokeh
	display_results(averaged_results)



main()

#
