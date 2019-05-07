import numpy as np
from numba import jit
from sklearn.metrics import accuracy_score


#Filter inputs based on performance as defined by the performance function given. By default it will use accuracy
class perf_filter(object):
	def __init__(self, performance_function = accuracy_score):
		self.performance_function = performance_function
		self.ranked_args = np.array([])

	def selection(self, truth, preds, performance_function = None):
		if not performance_function:
			performance_function = self.performance_function

		perfs = np.array([performance_function(truth, np.argmax(pred,axis = 1)) for pred in preds])
		
		self.ranked_args = np.argsort(perfs)[::-1]

		return self

	def filter(self, preds, nbr_to_filter = 3):

		return preds[self.ranked_args[:nbr_to_filter]]


@jit(nopython = True)
def calc_cognitive_diversity_matrix(ordered_preds, matrix, nbr_clf, nbr_events):
	#For each pair of clf calculate the ordered point by point difference and sum it up
	#Returns a set of pairs with their CD
	for ind1 in range(nbr_clf):
		for ind2 in range(nbr_clf):
			d=0.0
			for i in range(nbr_events):
				d = d + (ordered_preds[ind1,i] - ordered_preds[ind2,i])**2
			matrix[ind1,ind2] = (d/nbr_events)**(1/2)
#Filter inputs based on Cognitive diversity Strength
class cds_simple_filter(object):
	def __init__(self, performance_function):

		self.ranked_args = np.array([])

	def selection(self, truth, preds):
		#Get CDS for each clf
		cds = self.naive_cognitive_diversity_strength(self.cognitive_diversity_matrix(preds))
		self.ranked_args = np.argsort(cds)[::-1]
		return self

	def filter(self, preds, nbr_to_filter = 3):

		return preds[self.ranked_args[:nbr_to_filter]]

	#NEED TO RECEIVE ONE VALUE PER CLF PER EVENT (no 3d but 2d matrix)
	def cognitive_diversity_matrix(self,preds):
		#Order predictions from each clf
		ordered_preds = np.sort(preds, axis = - 1)
		matrix = np.zeros((ordered_preds.shape[0], ordered_preds.shape[0]))
		nbr_clf = ordered_preds.shape[0]
		nbr_events = ordered_preds.shape[1]
		calc_cognitive_diversity_matrix(ordered_preds, matrix, nbr_clf, nbr_events)
		return matrix

	#Simple summing
	def naive_cognitive_diversity_strength(self,cdm):

		return np.sum(cdm, axis = 1)

#Modified implementation of the cds paper to handle multi-class
#runs a std cds calc on each class and average them out.
class cds_filter(cds_simple_filter):
	def __init__(self, performance_function = None):

		cds_simple_filter.__init__(self,performance_function)

	def selection(self, truth, preds):
		#Get CDS for each clf
		cds = np.sum([self.naive_cognitive_diversity_strength(self.cognitive_diversity_matrix(preds[:,:,class_nbr])) for class_nbr in range(preds.shape[-1])], axis = 0)
		self.ranked_args = np.argsort(cds)[::-1]
		return self

	def filter(self, preds, nbr_to_filter = 3):

		return preds[self.ranked_args[:nbr_to_filter]]

#Filter inputs based on Cognitive diversity Strength and performance simultanously
class ruler_filter(object):
	def __init__(self, performance_function = accuracy_score):
		self.performance_function = performance_function
		self.perf = perf_filter(self.performance_function)
		self.cds = cds_filter(self.performance_function)
		self.perf_clf = []
		self.cds_clf = []
		self.ranked_args = np.array([])
		
	def selection(self, truth, preds, performance_function = None):
		#Train both cds and perf filter
		self.perf = self.perf.selection(truth, preds)
		self.cds = self.cds.selection(truth, preds)
		#Get ranked clf
		self.perf_clf = self.perf.ranked_args
		self.cds_clf = self.cds.ranked_args
		#Order clf using ruler method
		selected = []
		passed = []
		i = 0
		#Select the first x values in common
		for p, c in zip(self.perf_clf, self.cds_clf):
			i += 1
			if (p in passed or p == c) and not p in passed:
				selected.append(p)
			elif not p in passed:
				passed.append(p)

			if c in passed and not c in selected:
				selected.append(c)
			elif not c in passed:
				passed.append(c)

		self.ranked_args = np.array(selected)

		return self

	def filter(self, preds, nbr_to_filter = 3):

		return preds[self.ranked_args[:nbr_to_filter]]






#
