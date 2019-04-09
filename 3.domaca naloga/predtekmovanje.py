import lpputil
import linear
import numpy as np
import gzip
import csv
import datetime

class Predtekmovanje:
    def __init__(self):
        self.data = self.read_file("train_pred.csv.gz")
        self.test_data = self.read_file("test_pred.csv.gz")
        X = np.vstack(self.day_time(self.data))
        y = np.array(self.duration(self.data))
        test_matrix = np.vstack(self.day_time(self.test_data))
        lr = linear.LinearLearner(lambda_=1.)
        napovednik = lr(X,y)
        result = [napovednik(line) for line in test_matrix]
        fo = open("predtekmovanje_results.txt", "wt")
        for l, e in zip(result, self.test_data):
            fo.write(lpputil.tsadd(e[6],l)+"\n")

    def day_time(self, data):
        matrix = []
        for e in data:
            row = [0]*31
            start = lpputil.parsedate(e[6])
            if (start.month in (11,12)):
                row[start.isoweekday()+23] = 1
                row[start.hour] = 1
                matrix.append(row)

        return matrix

    def duration(self, data):
        y = []
        for e in data:
            start = lpputil.parsedate(e[6])
            if (start.month in (11,12)):
                y.append(lpputil.tsdiff(e[8], e[6]))
        return y

    def read_file(self, file_path):
        f = gzip.open(file_path, "rt", encoding="UTF-8")
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        data = [ d for d in reader ]
        return data

p = Predtekmovanje()