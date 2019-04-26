import numpy as np
from sklearn.metrics import accuracy_score

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
