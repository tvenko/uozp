import csv
from math import sqrt
from itertools import product
from itertools import combinations
import re
import copy


def avg(lst):
    """Calculates the average of the given list."""
    return sum(lst) / len(lst)


class Clustering:
    """A class that wll perform hierarchical clustering on the Eurovision voting."""
    linkages = {"single": min, "complete": max, "average": avg}

    def __init__(self, path_to_file, linkage="average"):
        f = open(path_to_file, "rt", encoding="latin1")
        reader = csv.reader(f)
        self.header = next(reader)[16:63]
        # 16 to 63 are countries and how they voted
        data_transposed = []
        for line in reader:
            data_transposed.append([float(v) if v else None for v in line[16:63]])
        self.data = [list(i) for i in zip(*data_transposed)]
        self.linkage = self.linkages[linkage]
        # Clusters, without join history
        # ToDo dict with country names instead of numbers
        self.clusters = []
        for i in range(0, len(self.data)):
            self.clusters.append([i])

    def join_clusters(self, tuple_cluster):
        """Join the two clusters into the self.clusters list"""
        c1 = tuple_cluster[0]
        c2 = tuple_cluster[1]
        self.clusters.remove(c1)
        self.clusters.remove(c2)
        self.clusters.append(c1 + c2)

    def alternating_nones(self, x, y):
        """Return true if the rows have alternating Nones for voting values"""
        for v1, v2 in zip(self.data[x], self.data[y]):
            if (v1 is not None) and (v2 is not None):
                return False
        return True

    def row_distance(self, row1, row2):
        """Distance between rows with indices row1 and row2."""
        diffs = [(x - y) ** 2 for x, y in zip(self.data[row1], self.data[row2])
                 if (x is not None) and (y is not None)]
        if len(diffs) > 0:
            return sqrt(sum(diffs) / len(diffs))
        else:
            pass

    def cluster_distance(self, cluster1, cluster2):
        """Distance between clusters with indices cluster1 and cluster2."""
        dists = []
        for x, y in list(product(cluster1, cluster2)):
            if self.alternating_nones(x, y):
                continue
            else:
                dists.append(self.row_distance(x, y))
        if len(dists):
            return self.linkage(dists)
        else:
            pass

    def closest_clusters(self):
        """Return two closest clusters."""
        return min((self.cluster_distance(*c), c)
                   for c in combinations(self.clusters, 2)
                   if self.cluster_distance(*c) is not None)

    def build_regexp(self, numbers):
        """Build a regular expression that will find the numbers in any amount of brackets around them"""
        exp = '\[*'
        for i in numbers:
            exp += str(i) + '\W*'
        exp += '\]*'
        return exp

    def run(self, limit=5):
        """Perform hierarchical clustering."""
        # ToDo kreiraj zgodovino zdruzevanj za izris
        joining = copy.copy(self.clusters)
        while len(self.clusters) > limit:
            next_join = self.closest_clusters()
            print("Next countries to join: "),
            self.num_to_country(next_join[1][0], type=0)
            print("     and  ")
            self.num_to_country(next_join[1][1], type=0)
            c1_index = self.locate_cluster(joining, next_join[1][0])
            c2_index = self.locate_cluster(joining, next_join[1][1])
            c1 = joining[c1_index]
            c2 = joining[c2_index]
            joining.pop(c1_index)
            if c1_index < c2_index:
                c2_index -= 1
            joining.pop(c2_index)
            joining.append([c1 + c2])
            self.join_clusters(next_join[1])

        print("======+ FINAL RESULTS +======")
        print("------* CLUSTERS *-------")
        [self.num_to_country(c) for c in self.clusters]
        print("------* HISTORY *-------")
        print(joining)
        print("======+ END +======")

    def num_to_country(self, cluster_num, type=1):
        """Convert the cluster index numbers to the corresponding country"""
        if type:
            for n in cluster_num:
                print(n, ": ", self.header[n])
            print("------------")
        else:
            for n in cluster_num:
                print(n, ": ", self.header[n], ", ", end="")
            print()

    def locate_cluster(self, joining, cluster):
        """Find the cluster in the joining list"""
        # Joining keep a history record of joins while
        # cluster only has the numbers that need to be
        # inside a joining element.
        regexp = self.build_regexp(cluster)
        strings = [str(el) for el in joining]
        cluster_index = [i for i, item in enumerate(strings) if re.search(regexp, item)]
        return cluster_index[0]


hc = Clustering("eurovision-final.csv")