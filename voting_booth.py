#

import numpy as np
from sklearn.metrics import accuracy_score

#NOT MULTILABEL VOTING SCHEME

class voting_booth(object):
	def __init__(self):
		pass

	#Preds are expected with a shape: [nbr_classifiers, nbr_events, nbr_classes]
	def vote(self,preds):
		for clf_i,clf in enumerate(preds):
			for event_i,event in enumerate(clf):
				preds[clf_i,event_i,preds[clf_i,event_i].argmax()] = 1
		preds[preds<1] = 0
		ballots = np.sum(preds,axis = 0)
		ballots = ballots.argmax(axis = -1)
		return ballots
