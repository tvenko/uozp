import os
from collections import Counter
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import math
from unidecode import unidecode


class LanguageClasstering:

    #seznam datotek
    l = ["slo.txt", "slv.txt", "src1.txt", "src3.txt", "src4.txt", "src5.txt", "spn.txt", "rus.txt", "ruw.txt", "pql.txt", "por.txt", "ger.txt", "eng.txt", "itn.txt",
         "czc.txt", "dns.txt", "nrn.txt", "mkj.txt", "ltn.txt", "hng.txt", "grk.txt", "frn.txt", "fin.txt", "dut.txt"]
    #slovar jezikov
    fullName = {"slo":"Slovascina", "slv":"Slovenscina", "src1":"Bosanscina", "src3":"Srbscina", "src4":"Bosanscina cirilica", "src5":"Srbscina cirilica", "spn":"Spanscina",
            "rus":"Ruscina", "ruw":"Beloruscina", "pql":"Polscina", "por":"Portugalscina", "ger":"Nemscina", "eng":"Anglescina", "itn":"Italijanscina", "czc":"Cescina",
            "dns":"Danscina", "nrn":"Norvescina", "mkj":"Makedonscina", "ltn":"Latinscina", "hng":"Madzarscina", "grk":"Grscina", "frn":"Francoscina", "fin":"Finscina", "dut":"Nizozemscina"}

    country = []
    izpis = []
    indx = []
    matrix = {}

    def __init__(self, filepath="human_rights/ready/", mode="dendrogram", jezikFile=None, k=3):
        for filename in os.listdir(filepath):
            if filename in self.l:
                f = str(open(filepath+filename, "rt", encoding="utf8").read())              #preberemo datoteko
                f = f.lower()                                                               #vse crke pretvorimo v male crke in odstranimo prehode v novo vrstico
                f = f.replace("\n", " ")
                f = unidecode(f)                                                            #dekodiramo iz unikoda
                self.matrix[filename.split(".")[0]] = dict(Counter(self.walk(f, k)))        #poiscemo stevilo vseh terk v besedilu

        #ustavrimo si vse potrebne sezname
        for i, e in enumerate(self.l):
            self.izpis.append(self.fullName[(e.split('.')[0])])
            self.country.append(e.split('.')[0])
            self.indx.append([i])

        #ce zelimo izrisati dendrogram
        if(mode == "dendrogram"):
            z = linkage(self.indx, metric=self.cosinus)                                     #naredimo hierarhicni clustering
            plt.title("jezik")
            dendrogram(z, color_threshold=1, labels=self.izpis, show_leaf_counts=True)      #ustvarimo dendrogram
            plt.show()                                                                      #izrisemo dendrogram

        #ce zelimo ugotoviti v kaksnem jeziku je besedilo
        elif (mode == "najdiJezik"):
            jezik = str(open(jezikFile, "rt", encoding="utf8").read())                      #preberemo datoteko
            jezik = jezik.lower()                                                           #vse crke pretvorimo v male crke in odstranimo prehode v novo vrstico
            jezik = jezik.replace("\n", " ")
            jezik = unidecode(jezik)
            self.jezikMatrix = dict(Counter(self.walk(jezik, k)))                           #prestejemo vse terke v danem besedilu
            x = self.cosinusJezik(self.jezikMatrix)
            print("Besedilo je v jeziku {} z verjetnostjo {}% lahko pa je tudi v jezikih {} ali {}"
                  .format(self.fullName[x[0][1]], ((1-x[0][0])*100).__round__(3), self.fullName[x[1][1]], self.fullName[x[2][1]]))


    def cosinus(self, index1, index2):
        """funkcija ki izracuna kosinusno razdaljo med dvema besediloma z indeksom index1 in index2"""
        key1, key2 = self.country[int(index1[0])], self.country[int(index2[0])]
        return self.cos(self.matrix[key1],self.matrix[key2], "dendrogram")

    def cosinusJezik(self, jezikMatrix):
        """funkcija, ki racuna kosinusno razdaljo med podanimi besedili in besedilom katerega jezik bi radi dolocili"""
        countries = []
        minimumList = [self.cos(self.matrix[e], self.jezikMatrix, "najdiJezik", e) for e in self.matrix]
        for i in range(3):
            countries.append(min(minimumList))
            minimumList.remove(min(minimumList))
        return countries                                                                    #vrnemo tri najbolj verjetne jezike z verjetnostmi

    def cos(self, m1, m2, mode, e=None):
        """funkcija za izracun kosinusne razdalje"""
        skalar = sum(m1[t]*m2[t] for t in set(m1).intersection(set(m2)))
        dist1 = math.sqrt(sum(x1**2 for x1 in m1.values()))
        dist2 = math.sqrt(sum(x1**2 for x1 in m2.values()))
        if (mode == "dendrogram"):
            return 1-skalar/(dist1*dist2)
        else:
            return 1-skalar/(dist1*dist2), e

    def walk (self, string, k):
        """funkcija ki nam ustvari terke dolzine k"""
        for i in range(len(string)-(k-1)):
            yield(string[i:(k+i)])

#LanguageClusstering(<filepath>, mode=<"dendrogram" ali "najdiJezik">, jezikFile=<filepath do jezika, ki ga zelimo dolociti>, k=<velikost terk>)
#lc = LanguageClasstering(mode="najdiJezik", jezikFile="test/eng.txt", k=4)
lc = LanguageClasstering()


