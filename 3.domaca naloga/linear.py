from collections import defaultdict
import numpy
from scipy.optimize import fmin_l_bfgs_b
import scipy.sparse as sp
import numpy as np
import lpputil


def append_ones(X):
    if sp.issparse(X):
        return sp.hstack((np.ones((X.shape[0], 1)), X)).tocsr()
    else:
        return np.hstack((np.ones((X.shape[0], 1)), X))

def hl(x, theta):
    """
    Napovej verjetnost za razred 1 glede na podan primer (vektor vrednosti
    znacilk) in vektor napovednih koeficientov theta.
    """
    return x.dot(theta)

def cost_grad_linear(theta, X, y, lambda_):
    #do not regularize the first element
    sx = hl(X, theta)
    j = 0.5*numpy.mean((sx-y)*(sx-y)) + 1/2.*lambda_*theta[1:].dot(theta[1:])/y.shape[0]
    grad = X.T.dot(sx-y)/y.shape[0] + numpy.hstack([[0.],lambda_*theta[1:]])/y.shape[0]
    return j, grad

def pretekli_mesec(e):
    row = [0]*31
    start = lpputil.parsedate(e[6])
    if (start.month in (11,12)):        #gledamo samo november
        row[start.isoweekday()+23] = 1  #dan v tednu
        row[start.hour] = 1             #ura v dnevu
    return row

def polinom5(e):
    row = [0]*10
    start = lpputil.parsedate(e[6])
    row[0] = start.hour/24
    row[1] = start.isoweekday()/7
    row[2] = (start.hour/24)**2
    row[3] = (start.isoweekday()/7)**2
    row[4] = (start.hour/24)**3
    row[5] = (start.isoweekday()/7)**3
    row[6] = (start.hour/24)**4
    row[7] = (start.isoweekday()/7)**4
    row[8] = (start.hour/24)**5
    row[9] = (start.isoweekday()/7)**5
    return row

def polinom8(e):
    row = [0]*16
    start = lpputil.parsedate(e[6])
    row[0] = start.hour/24
    row[1] = start.isoweekday()/7
    row[2] = (start.hour/24)**2
    row[3] = (start.isoweekday()/7)**2
    row[4] = (start.hour/24)**3
    row[5] = (start.isoweekday()/7)**3
    row[6] = (start.hour/24)**4
    row[7] = (start.isoweekday()/7)**4
    row[8] = (start.hour/24)**5
    row[9] = (start.isoweekday()/7)**5
    row[10] = (start.hour/24)**6
    row[11] = (start.isoweekday()/7)**6
    row[12] = (start.hour/24)**7
    row[13] = (start.isoweekday()/7)**7
    row[14] = (start.hour/24)**8
    row[15] = (start.isoweekday()/7)**8
    return row

def celo_leto(e):
    row = [0]*31
    start = lpputil.parsedate(e[6])
    row[start.isoweekday()+23] = 1  #dan v tednu
    row[start.hour] = 1             #ura v dnevu
    return row

class LinearLearner(object):

    def __init__(self, lambda_=0.0):
        self.lambda_ = lambda_

    def __call__(self, data):
        X, y = [], []
        for e in data:
            X.append(pretekli_mesec(e))
            y.append(lpputil.tsdiff(e[8], e[6]))
        X = numpy.vstack(X)
        y = numpy.array(y)
        X = append_ones(X)
        th = fmin_l_bfgs_b(cost_grad_linear, x0=numpy.zeros(X.shape[1]), args=(X, y, self.lambda_))[0]
        return LinearRegClassifier(th)

class LinearRegClassifier(object):

    def __init__(self, th):
        self.th = th

    def __call__(self, data):
        x = numpy.hstack(([1.], pretekli_mesec(data)))
        return hl(x, self.th)