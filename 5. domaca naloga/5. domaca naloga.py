import csv
import math
import numpy as np


# funkcija za branje podatkov iz datoteke
def read_data(file_path):
    f = open(file_path, "rt")
    reader = csv.reader(f, delimiter="\t")
    data = [d for d in reader]
    return data


class PreprostPriporocilni:
    """ Implementacija preprostega priporocilnega sistema, ki oddaja napovedi upostevajoc zgolj kvaliteto filma in
    zahtevnost uporabnika."""
    def __init__(self, file_path, file_path_test):
        self.data = read_data(file_path)
        self.testData = read_data(file_path_test)
        self.users = []                                 # vsi uporabniki
        self.movies = []                                # vsi filmi
        self.user_movies = {}                           # slovar vseh filmov dolocenega uporabnika
        self.averageMovies = {}                         # slovar povprecnih ocen filmov
        self.averageUsers = {}                          # slovar povprecnih ocen uporabnikovih filmov
        self.userMissing = {}                           # slovar filmov, ki mankajo uporabniku

    def __call__(self, *args, **kwargs):
        self.all_user_and_movies()
        self.average_movie()
        self.average_user()
        print("dolocanje ocen neocenjenim filmom uporabnikov")
        for user in self.users:
            self.userMissing[user] = self.user_missing(user)
        print("rmse: ", self.rmse())

    def user_missing(self, user):
        """funkcija, ki dopolni vse ocene neocenenih filmov uporabnika"""
        movie_missing = {}
        x = [[movie, (self.averageMovies[movie] + self.averageUsers[user]) / 2] for e in self.user_movies[user]
             for movie in self.movies if (e != movie)]
        for e in x:
             movie_missing[e[0]] = e[1]
        return movie_missing

    def all_user_and_movies(self):
        """funkcija za pridobivanje vseh uporabnikov, filmov in filmov uporabnikov"""
        print("pridobivam vse filme in uporabnike")
        for e in self.data:
            if e[0] not in self.users:
                self.users.append(e[0])
                self.user_movies[e[0]] = []
            if e[1] not in self.movies:
                self.movies.append((e[1]))
            self.user_movies[e[0]].append(e[1])

    def average_user(self):
        """funkcija ki vrne povprecno oceno uporabnikov"""
        print("ocenjujem zahtevnost uporabnikov")
        for user in self.users:
            x = [int(e[2]) for e in self.data if (e[0] == user)]
            self.averageUsers[user] = sum(x) / len(x)

    def average_movie(self):
        """funkcija, ki vraca povprecno oceno filma"""
        print("racunam povprecno oceno vseh filmov")
        for movie in self.movies:
            x = [int(e[2]) for e in self.data if (e[1] == movie)]
            self.averageMovies[movie] = sum(x) / len(x)

    def rmse(self):
        """rmse"""
        print("racunam RMSE")
        x = [(int(e[2]) - self.averageMovies[e[1]]) ** 2 for e in self.testData for user in self.users
             if (user == e[0] and e[1] in self.user_movies[user])]
        return math.sqrt(sum(x)) / len(x)


class PodobnostPriporocilni:
    """Implementiran priporocilni sistem, ki temelji na podobnosti, merjeni s kosinusno razdaljo, med filmi"""
    def __init__(self, file_path, file_path_test):
        self.data = read_data(file_path)
        self.testData = read_data(file_path_test)
        self.movies = []                            # vsi filmi
        self.users = []                             # vsi uporabniki
        self.user_movies = {}                       # slovar vseh filmov uporabnika
        self.user_scores = {}                       # slovar vseh ocen uporabnika
        self.movie_scores = {}                      # slovar vseh ocen filma
        self.movie_user_score = {}                  # slovar vseh uporabnikov, njihovih filmov in ocen le teh

    def __call__(self, *args, **kwargs):
        self.all_movies_users()
        print("dopolnjujem ocene uporabnikov")
        for user in self.users:
            print(user)
            self.user_missing(user)
        print("rmse: ", self.rmse())

    def user_missing(self, user):
        """funkcija, ki dopolni vse neocenjene filme uporabnikov"""
        print("filmi uporabnikov: ", len(self.user_movies[user]), " vsi filmi: ", len(self.movies))
        for movie in self.movies:
            suma = 0
            zmnozek = 0
            if movie not in self.user_movies[user]:
                x = [[self.cos(self.movie_user_score[movie], self.movie_user_score[user_movie]), int(self.user_scores[(user, user_movie)])]
                     for user_movie in self.user_movies[user]]
                for e in x:
                    suma += e[0]
                    zmnozek += e[0] * e[1]
                self.user_scores[(user, movie)] = zmnozek/suma

    def cos(self, movie, user_movie):
        """funkcija, ki vrne kosinusno razdaljo med dvema vektorjema ocen filmov"""
        skalar = sum([movie[k]*user_movie[k] for k in
                     set(movie.keys()).intersection(set(user_movie.keys()))])
        dist1 = math.sqrt(sum(int(x1) ** 2 for x1 in movie))
        dist2 = math.sqrt(sum(int(x1) ** 2 for x1 in user_movie))
        return 1 - skalar / (dist1 * dist2)

    def all_movies_users(self):
        """funkcija, ki vraca sezname vseh uporabnikov in filmov"""
        print("pridobivam podatke o vseh filmih in uporabnikih")
        for e in self.data:
            if e[0] not in self.users:
                self.users.append(e[0])
                self.user_movies[e[0]] = []
            if e[1] not in self.movies:
                self.movies.append((e[1]))
                self.movie_scores[e[1]] = []
                self.movie_user_score[e[1]] = {}
            self.user_movies[e[0]].append(e[1])
            self.user_scores[(e[0], e[1])] = e[2]
            self.movie_scores[e[1]].append(e[2])
            self.movie_user_score[e[1]][e[0]] = int(e[2])

    def rmse(self):
        """rmse"""
        print("racunam RMSE")
        x = [(int(e[2]) - self.user_scores[(e[0], e[1])]) ** 2 for e in self.testData if(e[1] in self.user_scores.keys()[1])]
        return math.sqrt(sum(x))/len(x)


class LatentniModel:
    """Implementacija priporocilnega sistema na podlagi latentnega modela"""
    def __init__(self, file_path, file_path_test, K):
        self.K = K                                                      # stopnja razcepa k
        self.data = read_data(file_path)
        self.testData = read_data(file_path_test)
        self.users = []                                                 # vsi uporabniki
        self.movies = []                                                # vsi filmi
        self.user_movie_score = {}                                      # slovar uporabnikov, njihovih filmov in ocen, ki so jim dodelili
        self.all_users_movies()
        # inicializavija matrik P in Q z vrednostmi med -0.01 in 0.01
        p = 0.02*np.random.random_sample((len(self.users), K))-0.01
        q = 0.02*np.random.random_sample((len(self.movies), K))-0.01
        self.P = {}
        self.Q = {}
        # iz matrik pretvorimo P in Q v slovarje
        for i, user in enumerate(self.users):
            self.P[user] = p[i]
        for i, movie in enumerate(self.movies):
            self.Q[movie] = q[i]

    def __call__(self, alfa = 0.01, ni = 0.01):
        print("racunam P in Q")
        for _ in range(30):
            for user in self.users:
                for movie in self.movies:
                    if (movie in self.user_movie_score[user].keys()):
                        # izracun cilje funkcije s regularizacijo
                        eui = (self.user_movie_score[user][movie] - self.P[user].dot(self.Q[movie]))
                        # popravimo P in Q upostevajoc gradient
                        for k in range(0,self.K):
                            self.P[user][k] += alfa * (eui * self.Q[movie][k] - ni * self.P[user][k])
                            self.Q[movie][k] += alfa * (eui * self.P[user][k] - ni * self.Q[movie][k])
        print("rmse: ", self.rmse())

    def all_users_movies(self):
        """funkcija, ki vraca sezname vseh uporabnikov in filmov"""
        print("pridobivam vse filme in uporabnike")
        for e in self.data:
            if e[0] not in self.users:
                self.users.append(e[0])
                self.user_movie_score[e[0]] = {}
            if e[1] not in self.movies:
                self.movies.append((e[1]))
            self.user_movie_score[e[0]][e[1]] = int(e[2])

    def rmse(self):
        """rmse"""
        print("racunam RMSE")
        x = [(int(e[2]) - (self.P[e[0]].dot(self.Q[e[1]]))) ** 2 for e in self.testData if (e[1] in self.Q.keys())]
        return math.sqrt(sum(x)) / len(x)

# razred za preprost priporocilni sistem.
# p = PreprostPriporocilni("movielens-100k-train.tab", "movielens-100k-test.tab")

# razred za priporocilni sistem, ki temelji na podobnosti.
# p = PodobnostPriporocilni("movielens-100k-train.tab", "movielens-100k-test.tab")

# razred za priporocilni sistem, ki temelji na latentnih modelih.
p = LatentniModel("movielens-100k-train.tab", "movielens-100k-test.tab", 1)
p()
