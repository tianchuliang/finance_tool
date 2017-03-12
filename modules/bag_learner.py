import numpy as np

class BagLearner(object):

    def __init__(self, learner = None, kwargs = None, bags = 20, boost = False, verbose = False):
        self.verbose = verbose 
        self.bags = bags
        self.boost = boost 
        self.kwargs = kwargs
        self.learners = [learner(**kwargs) for i in range (0, self.bags)]

    def addEvidence(self, dataX, dataY):
        for learner in self.learners:
            sample_indices = np.random.choice(dataX.shape[0], dataX.shape[0])
            learner.addEvidence(dataX[sample_indices],dataY[sample_indices])
        
    def query(self,points):
        from scipy.stats import mode
        # predictions = np.array([learner.query(points) for learner in self.learners])
        predictions = []
        for learner in self.learners:
            predictions.append(learner.query(points))
        predictions = np.array(predictions)
        result = mode(predictions, axis = 0)[0][0]
        if self.verbose:
            print 'predictions: ', np.array(result)
        return np.array(result)