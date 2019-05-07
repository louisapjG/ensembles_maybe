"""
Louis Gobin
Mixed Group Scores implementation. For references.
Setting stage for a MGR (mixed group ranks) implementation
"""
import itertools
import numpy as np
from sklearn.metrics import accuracy_score
from scipy.stats import rankdata
from joblib import Parallel, delayed,  Memory

"""
MGS
fit:
	Assign perfs to incoming clf
	for n in range(nbr_clfs): fit a box each
predict:
	for all boxes run predict.
	return merged all returned preds by avg.
"""
class MGS(object):
	def __init__(self, score_function = accuracy_score, n_jobs = 1):
		self.boxes=[]
		self.score_function = accuracy_score
		self.n_jobs = n_jobs

	#In: X: np.array X.shape = (nbr_info_sources, nbr_events, nbr_classes)
	#In: y: np.array y.shape = (nbr_events)
	def fit(self, Xs, y):
		#Assign ranks to incoming preds
		performances = np.array([self.score_function(y,np.argmax(X,axis = 1)) for X in Xs])
		#Create and fit all the boxes
		if(self.n_jobs == 1 or self.n_jobs == 0):
			self.boxes = [BOX(nbr_clf_per_merge = n+1, score_function = self.score_function).fit(Xs, y, performances) for n in range(Xs.shape[0])]
		else:
			self.boxes = Parallel(n_jobs=self.n_jobs,verbose=0)(delayed (self.fit_multi)(n, Xs, y, performances) for n in reversed(range(Xs.shape[0])))
		
		return self

	def fit_multi(self,n,Xs,y,performances):
		box = BOX(nbr_clf_per_merge = n+1, score_function = self.score_function)
		return box.fit(Xs, y, performances)
	
	def predict_proba(self, Xs):
		return np.sum([box.predict_proba(Xs) for box in self.boxes],axis = 0) / len(self.boxes)
"""
BOX
fit:
	RAssign ranks of each input performance. HIGHER IS BETTER
	Apply weights to each set of scores: normalize by dividing by highest weight
	Create all combinations of len nbr_clf_per_merge
	Evaluate all combinations, select the best one. Remember it.
predict:
	Apply remembered weights. Execute remembered merge. Return it
"""
class BOX(object):
	def __init__(self,nbr_clf_per_merge=1, score_function = accuracy_score):
		self.nbr_clf_per_merge=nbr_clf_per_merge
		self.score_function = score_function

		self.best_combination_index = None
		self.best_weights = None

	def fit(self, Xs, y, performances = np.array([])):#,nbr_clf_per_merge = 0):
		# if(nbr_clf_per_merge != 0):
		# 	self.nbr_clf_per_merge = nbr_clf_per_merge
		if performances.shape[0] == 0:
			#Assign ranks to incoming preds
			performances = np.array([self.score_function(y,np.argmax(X,axis = 1)) for X in Xs])

		#Create all combinations of len nbr_clf_per_merge
		combination_indexes = itertools.combinations(list([a for a in range(Xs.shape[0])]),self.nbr_clf_per_merge)
		best = 0.
		for combination_index in combination_indexes:
			sub_Xs = Xs[combination_index,:,:]
			sub_perf = performances[np.array(combination_index)]
			#Sort small to high and assign weights from len to 1
			if (len(combination_index) == 1):
				#Only one element mix, get best from performances
				self.best_combination_index = list(combination_indexes)[np.argmax(performances)]
				self.best_weights = [1]
				break
			weights = len(sub_perf)+1 - rankdata(sub_perf)
				
			#Apply weights and normalize
			weighted_Xs = np.multiply(sub_Xs.T, weights).T / len(sub_perf)
			#Merge
			preds = np.sum(weighted_Xs, axis = 0) / weighted_Xs.shape[0]
			#Evaluate
			perf = self.score_function(y,np.argmax(preds,axis = 1))
			if(perf >= best):
				best = perf
				self.best_combination_index = combination_index
				self.best_weights = weights
		return self
	
	def predict_proba(self, Xs):
		#Weight and normalize
		weighted_Xs = np.multiply(Xs[self.best_combination_index,:,:].T, self.best_weights).T / len(self.best_combination_index)
		#Merge and normalize
		return np.sum(weighted_Xs, axis = 0) / weighted_Xs.shape[0]
		
