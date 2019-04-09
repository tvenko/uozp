# logisticna_regresija

1. (30 točk) Implementirajte logistično regresijo. Uporabite priloženo ogrodje ter ga dopolnite z:

  izračunom verjetnosti za posamezni primer, h,
  cenilno funkcijo, cost, in
  gradientom cenilne funkcije, grad.
  Opazili boste, da sta gradnja napovednega modela in napovedovanje razreda posameznim primerom ločeni: razred LogRegLearner iz učnih podatkov zgradi napovedni model tipa LogRegClassifier, ki lahko nato za poljuben vektor značilk napove verjetnosti obeh razredov. Ogrodje rešuje optimizacijski problem s funkcijo fmin_l_bfgs_b (preprost primer uporabe).

  Preverjanje:

  S testi posameznih funkcij. Ker vemo, da je lahko cost in grad delimo in množimo s poljubno konstanto, prilagodite vašo rešitev testom. Najverjetneje bo treba deliti s številom primerov. Datoteko s testi shranite v direktorij z vašo rešitvijo in popravite import.
  Če gradnjo modela logistične regresije brez regularizacije poženete na celotnih podatkih reg.data, vam mora zgrajen model vse primere uvrstiti prav tako, kot je zapisano v reg.data.
2. (20 točk) Izdelani logistični regresiji dolepite kodo za izris napovedi. Na celotnem podatkovnem naboru reg.data (brez regularizacije) dobite spodnji izris (če ga ne, popravite logistično regresijo), kjer točke označujejo primere posameznih razredov, barva ozadja pa verjetnost napovedi v tistem delu prostora. Na abscisi je vrednost prve značilke (indeks 0), na ordinati pa druge (indeks 1).
  Raziščite vpliv regularizacije na napovedi, tako da spreminjate vrednosti parametra lambda. V poročilo priložite izrise za tri vam zanimive stopnje regularizacije. Katera se vam zdi najboljša in zakaj?

3. (25 točk) Implementirajte k-kratno prečno preverjanje kot funkcijo test_cv(learner, X, y, k=5), ki vrne napovedi za vse primere v enakem zaporedju kot so v X, le da nikoli ne uporablja istih primerov za napovedi in učenje. Razvijte še mero napovedne točnosti kot funkcijo CA(real, predictions).
  Poročajte o točnosti za širok nabor vrednosti lambda, kjer točnost merite:

  s 5-kratnim prečnim preverjanjem in (funkcija test_cv) in
  z gradnjo modela in napovedovanjem istih (vseh) primerov (funkcija test_learning).
  Preverjanje:

  S testi za CA in test_cv. 
4. (25 točk) Ustvarite dve poljubni skupini slik, recimo skupino stolov in skupino miz (vsaj 15 v vsaki skupini; poskrbite za to, da dodate tudi slike, za katere se vam zdi, da je ločevanje med razredoma težje). Sedaj jih z logistično regresijo poskusite ločevati. Za to uporabite orodje Orange, kjer slike z ImageNet najprej pretvorite v značilke. Katera vrednost stopnje regularizacije je najbolj primerna za vaše slike? Preglejte širok razpon vrednosti, uporabite prečno preverjanje in poročajte o rezultatih. Pri katerih slikah se vaš model obnese najslabše?

  Oddaja: Oddajte poročilo in izvorno kodo (kot pri prejšnji nalogi). Poročilo mora biti napisano s predpisano predlogo.

  Pri domači nalogi boste potrebovali knjižnjice numpy, scipy in matplotlib.
