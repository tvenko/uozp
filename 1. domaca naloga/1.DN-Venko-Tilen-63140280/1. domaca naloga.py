from math import sqrt
from itertools import combinations
from itertools import product
from collections import Iterable
import csv

def avg(lst):
    return sum(lst) / len(lst)

def colomn(matrix, i):
    return[row[i] for row in matrix]

class Clustering:
    linkages = {"single": min, "complete": max, "average": avg}

    def __init__(self, file_name, linkage="average"):
        f = open(file_name, "rt", encoding="latin1")
        self.t = [[i for i in l] for l in csv.reader(f)]                           #preberemo podatke v tabelo
        self.header = self.t[0]                                                    #si zapomnimo drzave v header
        del self.header[:16]
        self.country = []
        self.data = []
        self.dataArray = []
        self.clusters = []
        self.drzave = {}
        for i in range(1,len(self.t)):
            self.country.append(self.t[i][1])
            self.data.append([float(v) if v else None for v in self.t[i][16:]])    #naredimo tabelo podatkov v katero na prazna mesta vpisemo None
        self.linkage = self.linkages[linkage]
        for i in range(len(self.data[0])):
            self.dataArray.append(colomn(self.data, i))                            #obrnemo tabelo podatkov
            self.clusters.append([i])
            self.drzave.update({i: self.header[i]})                                #naredimo slovar drzav

    def column_distance(self, r1, r2):
        """Evklidska razdalja med dvema stolpcema drzav"""
        euklid = [(x - y) ** 2
                  for x, y in zip(r1, r2)
                  if (x is not None) and (y is not None)]
        if (len(euklid) > 0):
            return sqrt((sum(euklid))/len(euklid))
        else:
            return 999


    def cluster_distance(self, c1, c2):
        """Razdalja med dvema clustroma"""
        l = []
        for a,b in product(c1,c2):                                                  #iteriramo cez vse pare drzav v clustrih c1 in c2
            l.append(self.column_distance(self.dataArray[a], self.dataArray[b]))
        return self.linkage(l)                                                      #vrnemo najkrajso razdaljo glede na vrednost v self.linkage


    def closest_clusters(self):
        """Vrne najblizja clustra"""
        # 1 vrstica, nekaj podobnega temu spodaj
        dist, d = min((self.cluster_distance(*c), c)                                #iteriramo cez vse pare clustroc in poisemo minimalni par
                      for c in combinations(self.clusters, 2))
        return d                                                                    #vrne par clustrov ki sta najblizja



    def run(self, st_clustrov):
        """Izvajanje hierarhicnega clusteringa"""
        while len(self.clusters) > st_clustrov:
            pair = self.closest_clusters()
            self.clusters = [x for x in self.clusters if x not in pair] + [pair[0] + pair[1]]
        for l in self.clusters:
            for c in l:
                print(self.drzave[c])
            print("************")


hc = Clustering("eurovision-final.csv")
hc.run(3)
