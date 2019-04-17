#Class
import numpy as np
import itertools
from scipy.stats import rankdata

class MGR(object):
	def __init__(self):
		self.boxes=[]

	#Takes in j*n array with j: number of classifers, n: number of events
	def fit(self,data_in,truth,score_function,perf_at=[],perf_coef=[]):
		#Create j boxes: nbr merged classifiers 1 to j
		for box_nbr in range(data_in.shape[0]):
			box=BOX(nbr_clf_per_merge=box_nbr+1)
			#In box 1: Rank classsifiers from 1 to j
			if(box_nbr==0):
				data_in_ranks=box.rank_classifiers(data_in,truth,score_function,perf_at,perf_coef)
			#box fit
			box.fit(data_in,truth,clf_ranking=data_in_ranks,score_function=score_function,perf_at=perf_at,perf_coef=perf_coef)
			self.boxes.append(box)

	#Takes in j*n array with j: number of classifers, n: number of events
	def predict(self,data_in):
		predictions=[]
		for box in self.boxes:
			#Run data through boxes
			predictions.append(box.predict(data_in))
		#Sum up the best classifiers of each box
		summed_ranks=np.sum(np.array(predictions),axis=0)#/np.array(predictions).shape[0]
		inds=summed_ranks.argmax(axis=1)
		ranks=rankdata(summed_ranks.max(axis=1),method="ordinal")
		#Returns score! not rank!
		norm_ranks = (ranks-np.min(ranks))/(np.max(ranks)-np.min(ranks))
		tbr=np.zeros_like(summed_ranks)
		tbr[:,inds[0]]=norm_ranks
		return tbr

class BOX(object):
	def __init__(self,nbr_clf_per_merge=1):
		self.nbr_clf_per_merge=nbr_clf_per_merge
		self.best_combiner_ind=0
		self.combination_indexes=[]
		self.clf_ranking=[]

	def fit(self,data_in,truth,score_function,clf_ranking=[],perf_at=[],perf_coef=[]):
		if(clf_ranking==[]):
			clf_ranking=self.rank_classifiers(data_in)
		self.clf_ranking=clf_ranking
		#For each subsequent box: use ranks from box 1 to do combinations: ie: box3: ABC=(rankA*rankd1_inA+rankB*rankd1_inB+rankC*rankd1_inC)/rankA+rankB+rankC
		combination_indexes = list(itertools.combinations(list([a for a in range(len(data_in))]),self.nbr_clf_per_merge))
		self.combination_indexes=combination_indexes
		predictions=[]
		for indexes in combination_indexes:
			data_selected=data_in[np.array(indexes)]
			clf_ranks_selected=clf_ranking[np.array(indexes)]
			weighted_data=(np.array(data_selected).T*np.array(clf_ranks_selected)).T
			if(len(indexes)>1):
				predictions.append(np.sum(weighted_data,axis=0)/np.sum(clf_ranks_selected))
			else:
				predictions=weighted_data

		#For all boxes rank the classifiers inside it and remember the best one
		ranks=self.rank_classifiers(data_in=np.array(predictions),truth=truth,score_function=score_function,perf_at=perf_at,perf_coef=perf_coef)
		self.best_combiner_ind=list(ranks).index(min(ranks))


		#return self

	def predict(self,data_in):
		data_selected=np.array(data_in[np.array(self.combination_indexes[np.array(self.best_combiner_ind)])])
		clf_ranks_selected=np.array(self.clf_ranking[np.array(self.combination_indexes[np.array(self.best_combiner_ind)])])
		weighted_data=(np.array(data_selected).T*clf_ranks_selected).T
		#Normalize
		score=np.sum(weighted_data,axis=0)
		norm_score=(score-np.min(score))/(np.max(score)-np.min(score))
		return norm_score

	def rank_classifiers(self,data_in,truth,score_function,perf_at=[],perf_coef=[]):
		#For every column use score function to assess it
		perfs=[]
		for clf in data_in:
			perfs.append(score_function(truth,clf,perf_at=perf_at,perf_coef=perf_coef))
		#Rank the scores given to each column
		ranked_perf=rankdata(perfs)
		#return Ranks
		return ranked_perf



