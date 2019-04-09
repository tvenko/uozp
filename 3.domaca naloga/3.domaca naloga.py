import gzip
import numpy as np
import csv
import lpputil
import linear
import math

from collections import defaultdict

def linekey(d):
    return tuple(d[2:5])

class SeparateBySetLearner(object):

    def __init__(self, base):
        self.base = base

    def __call__(self, data):
        rsd = defaultdict(list)
        rsc = {}

        #locimo razlicne avtobusne linije
        for d in data:
            rsd[linekey(d)].append(d)

        #zgradi model za vsako linijo
        for k in rsd:
            cl = self.base(rsd[k])
            rsc[k] = cl
        return SeparateBySetClassifier(rsc)

class SeparateBySetClassifier(object):

    def __init__(self, classifiers):
        self.classifiers = classifiers

    def __call__(self, x):
        #zarad cudnih koncnih postaj je tole treba
        try:
            return self.classifiers[linekey(x)](x)
        except:
            return np.mean([c(x) for c in self.classifiers.values()])

class RMSE:
    #vrne RMSE med pdatki, ki smo jih pridelali s linearno regresijo in realnimi podatki
    def __init__(self, test, real):
        self.test = test
        self.real = real

    def __call__(self):
        return math.sqrt(sum([lpputil.tsdiff(e1, e2)**2 for e1,e2 in zip(self.test, self.real)])/len(self.test))

class MAE:
    #vrne MAE med pdatki, ki smo jih pridelali s linearno regresijo in realnimi podatki
    def __init__(self, test, real):
        self.test = test
        self.real = real

    def __call__(self):
        return sum([abs(lpputil.tsdiff(e1, e2)) for e1, e2 in zip(self.test, self.real)])/len(self.test)

def loci_po_mesecu(data):
    #izlocimo mesec november, da lahko na njem izvajamo interno testiranje
    test_data, learn_data, real = [], [], []
    for e in data:
        if (lpputil.parsedate(e[6]).month > 10):
            test_data.append(e)
            real.append(lpputil.parsedate(e[8]))    #zapomnimo si dejanski cas prihoda avtibusa na koncno postajo
        else:
            learn_data.append(e)
    return learn_data, test_data, real

def read_file(file_path):
    #funkcija za branje podtkov iz datoteke
    f = gzip.open(file_path, "rt", encoding="UTF-8")
    reader = csv.reader(f, delimiter="\t")
    next(reader)                                    #preskocimo glavo tabele
    data = [d for d in reader]
    return data

if __name__ == "__main__":

    #preberemo datoteke ankaterih se ucimo in tiste na katerih testiramo
    data = read_file("train.csv.gz")
    test_data = read_file("test.csv.gz")

    #zgradimo model
    l = SeparateBySetLearner(linear.LinearLearner(lambda_=1.))
    c = l(data)

    fo = open("results.txt", "wt")
    for l in test_data:
        fo.write(lpputil.tsadd(l[-3], c(l)) + "\n")
    fo.close()

    #preverjamo na internih podatkih
    data, test_data, real = loci_po_mesecu(data)

    l = SeparateBySetLearner(linear.LinearLearner(lambda_=1.))
    c = l(data)
    results = []
    for l in test_data:
        results.append(lpputil.tsadd(l[-3], c(l)))
    rmse = RMSE(results, real)
    mae = MAE(results, real)
    print("RMSE: ")
    print(rmse())
    print("MAE: ")
    print(mae())