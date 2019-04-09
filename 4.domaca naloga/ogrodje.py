import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from math import log
from math import e
from math import sqrt
import unittest

def load(name):
    """
    Open the file. Return a data matrix X (columns are features)
    and a vector of classes.
    """
    data = np.loadtxt(name)
    X, y = data[:,:-1], data[:,-1].astype(np.int)
    return X,y

def h(x, theta):
    """ 
    Predict the probability for class 1 for the current instance
    and a vector theta.
    """
    # ... your code
    return 1/(1+e**(-x.dot(theta)))

def cost(theta, X, y, lambda_):
    """
    Return the value of the cost function. Because the optimization algorithm
    used can only do minimization you will have to slightly adapt equations from
    the lectures.
    """
    reg = lambda_ * sum([e**2 for e in theta])
    return -1/len(y)*sum([yi*log(max(h(x, theta), 0.1**15)) + (1 - yi)*log(max((1 - h(x, theta)), 0.1**15)) for x, yi in zip(X, y)])+reg

def grad(theta, X, y, lambda_):
    """
    The gradient of the cost function. Return a numpy vector of the same
    size at theta.
    """

    l = []
    for i,e in enumerate(theta):
        l.append(1/len(y)*sum([(h(x,theta)-yi)*x[i] for x, yi in zip(X,y)])+2*lambda_*e)

    return np.array(l)

def CA(real, predictions):
    """return RMSE"""
    return sqrt(sum([(e1-e2)**2 for e1, e2 in zip(real, predictions)]) / len(real))

class LogRegClassifier(object):

    def __init__(self, th):
        self.th = th

    def __call__(self, x):
        """
        Predict the class for a vector of feature values.
        Return a list of [ class_0_probability, class_1_probability ].
        """
        x = np.hstack(([1.], x))
        p1 = h(x, self.th)
        return [ 1-p1, p1 ] 

class LogRegLearner(object):

    def __init__(self, lambda_=0.0):
        self.lambda_ = lambda_

    def __call__(self, X, y):
        """
        Build a prediction model for date X with classes y.
        """
        X = np.hstack((np.ones((len(X),1)), X))

        #optimization as minimization
        theta = fmin_l_bfgs_b(cost,
            x0=np.zeros(X.shape[1]),
            args=(X, y, self.lambda_),
            fprime=grad)[0]

        return LogRegClassifier(theta)

if __name__ == "__main__":
    #
    # Usage example
    #

    X,y = load('reg.data')

    learner = LogRegLearner(lambda_=0.0)
    classifier = learner(X,y) # we get a model

    prediction = classifier(X[0]) # prediction for the first training example
    print(prediction)
