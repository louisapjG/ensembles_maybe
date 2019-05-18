import numpy as np
import copy
import itertools
from datetime import datetime
from joblib import Parallel, delayed,  Memory
from shutil import rmtree
from sklearn import svm, ensemble, linear_model, naive_bayes, discriminant_analysis, gaussian_process, neighbors, tree
from sklearn.cluster import KMeans, FeatureAgglomeration
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tempfile import mkdtemp

class board_opinion(object):
	def __init__(self, n_jobs = 1,time_serie = False, nbr_train_test_split = 2, scoring = "accuracy", variabilities = [(None,None)], normalizations = [(None,None)], dim_red = [(None,None)], clf_params = [(ensemble.RandomForestClassifier(),{'n_estimators':[300]}),]):
		#List of variability cleanups methods
		# self.variabilities=[(VarianceThreshold(),{'threshold':[.1]}),
		# 					(None,None),
		# 					]
		self.variabilities = variabilities


		#List of Normalizations methods
		#normalizations=['MinMaxScaler','None']
		# self.normalizations = [(MinMaxScaler(),{'feature_range':[(0, 1)]}),
		# 						(None,None),
		# 						]
		self.normalizations = normalizations
		
		#List of Dimensions reduction methods
		# self.dim_red = [(PCA(),{'n_components':[0.6]}),
		# 				(FeatureAgglomeration(),{'n_clusters':[3]}),
		# 				(KMeans(),{'n_clusters':[3]}),
		# 				(None,None),
		# 				]
		self.dim_red = dim_red
		
		#List of clf
		# self.clf_params = [(svm.SVC(probability=True),{'C': [1], 'gamma': [0.7], 'kernel': ['rbf']}),
		# 					(svm.SVC(probability=True),{'C':[1],'kernel':['linear']}),
		# 					(ensemble.RandomForestClassifier(),{'n_estimators':[300]}),
		# 					(ensemble.AdaBoostClassifier(),{'base_estimator':[ensemble.RandomForestClassifier(n_estimators = 100),],'n_estimators':[100]}),
		# 					##(tree.DecisionTreeClassifier(),[{'criterion':["gini"]}]),	
		# 					(neighbors.KNeighborsClassifier(),{'n_neighbors':[7]}),
		# 					#MCONSUMPTION ON LARGET SETS(gaussian_process.GaussianProcessClassifier(),[{'random_state':[3]}]),
		# 					(ensemble.GradientBoostingClassifier(),{'loss':['deviance'],'n_estimators':[200],'max_depth':[4]}),
		# 					(ensemble.ExtraTreesClassifier(),{'n_estimators':[50]}),
		# 					##(naive_bayes.GaussianNB(),[{'priors':[None]}]),
		# 					#Linear classifier
		# 					#Naive Bayes
		# 					#Decision tree
		# 					]
		self.clf_params = clf_params

		#Dico of scores by opinions:
		self.opinions_acc={}
		self.opinions_algos={}

		self.n_jobs=n_jobs
		self.time_serie = time_serie
		self.nbr_train_test_split = nbr_train_test_split
		self.scoring = scoring

	def fit(self,data_in,data_target, predict_training_probas = False):
		#SPLIT INTO DATA_IN_/DATA_TARGET_ AND DATA_TEST AND TEST_TARGET
		list_returned=[]
		#List all combinations of options of opinions
		#Scikit-learn: Opinion packet
		opinions_combinations_options=list(itertools.product(self.variabilities,self.normalizations,self.dim_red,self.clf_params))
		if(self.n_jobs==1):
			#For all combinations:
			for c,combination in enumerate(opinions_combinations_options):
				list_returned.append(self.multi_fit(c,combination,data_in,data_target, predict_training_probas))
		else:
			#"multiprocessing" "threading"
			list_returned=Parallel(n_jobs=self.n_jobs, verbose=0, max_nbytes=1e4)(delayed (self.multi_fit)(c,combination,data_in,data_target, predict_training_probas) for c,combination in enumerate(opinions_combinations_options))

		best_opinion_acc=0
		training_probas = {}
		for ret in list_returned:
			c=ret[0]
			self.opinions_acc[c]=ret[1]
			self.opinions_algos[c]=ret[2]
			if(self.opinions_acc[c]>best_opinion_acc):
				best_opinion_acc=self.opinions_acc[c]
			if(predict_training_probas):
				training_probas[c] = ret[3]
				ys = ret[4]

		if(predict_training_probas):
			training_probas_lst = [training_probas[key] for key in sorted(training_probas.keys())]
			return np.array(training_probas_lst), np.array(ys)
		else:
			return self

	def multi_fit(self,c,combination,data_in,data_target, predict_training_probas = False):
		start = datetime.now()
		#Train all opinions (get scores, acc)
		cc = combination[3]
		clf,clf_params = cc[0],cc[1]
		var = combination[0]
		var,var_params = var[0],var[1]
		norm = combination[1]
		norm,norm_params = norm[0],norm[1]
		dim_red = combination[2]
		dim_red,dim_red_params = dim_red[0], dim_red[1]

		# print(dim_red)
		# print(norm)
		# print(var)
		# print(clf)

		# cachedir = mkdtemp()
		# memory = Memory(cachedir=cachedir, verbose=0)
		#Define pipeline
		#No norm + var + dim red
		if(var == None and norm == None and dim_red == None):
			pipe = Pipeline(steps = [
									('clf',clf)
									],
									#memory = memory
									)
			param_grid = {'clf__'+k:clf_params[k] for k in clf_params}
		#No var + norm
		elif(var == None and norm == None):
			pipe = Pipeline(steps = [
									('dim_red',dim_red),
									('clf',clf)
									],
									#memory = memory
									)
			param_grid = {'clf__'+k:clf_params[k] for k in clf_params}
			param_grid.update({'dim_red__'+k:dim_red_params[k] for k in dim_red_params})
		#No var + dim red
		elif(var == None and dim_red == None):
			pipe = Pipeline(steps = [
									('norm',norm),
									('clf',clf)
									],
									#memory = memory
									)
			param_grid = {'clf__'+k:clf_params[k] for k in clf_params}
			param_grid.update({'norm__'+k:norm_params[k] for k in norm_params})
		#No norm + dim red
		elif(norm == None and dim_red == None):
			pipe = Pipeline(steps = [
									('var',var),
									('clf',clf)
									], 
									#memory = memory
									)
			param_grid = {'clf__'+k:clf_params[k] for k in clf_params}
			param_grid.update({'var__'+k:var_params[k] for k in var_params})
		#No var
		elif var == None:
			pipe = Pipeline(steps = [
									('norm',norm),
									('dim_red',dim_red),
									('clf',clf)
									], 
									#memory = memory
									)
			param_grid = {'clf__'+k:clf_params[k] for k in clf_params}
			param_grid.update({'dim_red__'+k:dim_red_params[k] for k in dim_red_params})
			param_grid.update({'norm__'+k:norm_params[k] for k in norm_params})
		#No norm
		elif norm == None:
			pipe = Pipeline(steps = [
									('var',var),
									('dim_red',dim_red),
									('clf',clf)
									], 
									#memory = memory
									)
			param_grid = {'clf__'+k:clf_params[k] for k in clf_params}
			param_grid.update({'dim_red__'+k:dim_red_params[k] for k in dim_red_params})
			param_grid.update({'var__'+k:var_params[k] for k in var_params})
		#No dim red
		elif dim_red == None:
			pipe = Pipeline(steps = [
									('var',var),
									('norm',norm),
									('clf',clf)
									], 
									#memory = memory
									)
			param_grid = {'clf__'+k:clf_params[k] for k in clf_params}
			param_grid.update({'norm__'+k:norm_params[k] for k in norm_params})
			param_grid.update({'var__'+k:var_params[k] for k in var_params})
		#All in
		else:
			pipe = Pipeline(steps = [
									('var',var),
									('norm',norm),
									('dim_red',dim_red),
									('clf',clf)
									], 
									#memory = memory
									)
			param_grid = {'clf__'+k:clf_params[k] for k in clf_params}
			param_grid.update({'dim_red__'+k:dim_red_params[k] for k in dim_red_params})
			param_grid.update({'norm__'+k:norm_params[k] for k in norm_params})
			param_grid.update({'var__'+k:var_params[k] for k in var_params})

		#Fit pipeline
		if(self.time_serie):
			pipe = GridSearchCV(pipe, param_grid, scoring = self.scoring, cv = TimeSeriesSplit(n_splits = self.nbr_train_test_split), return_train_score = False)
		else:
			pipe = GridSearchCV(pipe, param_grid, scoring = self.scoring, cv = self.nbr_train_test_split, return_train_score = False)
		pipe=pipe.fit(data_in,data_target)
		# rmtree(cachedir)

		#Return training preds
		if(predict_training_probas):
			#Split data in repeatable pattern
			if(self.time_serie):
				splitter = TimeSeriesSplit(n_splits = self.nbr_train_test_split)
			else:
				splitter = StratifiedKFold(n_splits = self.nbr_train_test_split, random_state = 13)
			preds = []
			ys = []
			for train, test in splitter.split(data_in,data_target):
				X_train, y_train = data_in[train], data_target[train]
				X_test, y_test = data_in[test], data_target[test]
				retrained_est = pipe.best_estimator_.fit(X_train,y_train)
				preds.extend(retrained_est.predict_proba(X_test))
				ys.extend(y_test)

				# print(datetime.now() - start)
				# print()

			return c,pipe.best_score_,pipe,preds,ys
		
		else:
			return c,pipe.best_score_,pipe

	def predict_probas(self,data_in):
		#Apply opinion
		scores = {}
		for key in sorted(self.opinions_algos.keys()):
			op=self.opinions_algos[key]
			scores[key]=op.predict_proba(data_in)

		x=np.array([scores[a] for a in scores])
		
		return x
	
#
