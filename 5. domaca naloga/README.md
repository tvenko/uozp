# priporocilni_sistemi
V domači nalogi sem analiziral nabor podatkov o ocenah filmov Movielens 100k

1. (25%) Implementirajte preprost priporočilni sistem, ki oddaja napoved upoštevajoč zgolj kvaliteto filma (ali je v povprečju dobro ocenjen) in radodarnost uporabnika (ali v poprečju daje dobre ali slabe ocene). Sistem mora delovati tudi, če testnega uporabnika ali testnega filma ni bilo med učnimi primeri (predlagajte, kako boste ta problem rešili). Poročajte o uspešnosti te tehnike pri napovedi na testnih podatkih (RMSE).

2. (30%) Implementirajte priporočilni sistem, ki temelji podobnosti med filmi. Podobnost merite s kosinusno podobnostjo. Poročajte o RMSE na testnih podatkih.

3. (30%) Implementirajte priporočilni sistem na podlagi latentnega modela (metode ISMF oziroma njene variante z regularizacijo RISMF). Poročajte u uspešnosti te metode na testnih podatkih. Kako je uspešnost odvisna od ranga modela (število latetnih faktorjev k)?

4. (15%) Preverite, če so latentni faktorji smiselni: poženete algoritem za k=2 latentna faktorja, nato pa 20 filmov, ki jih v podatkih prepoznate in se vam zdijo zanimivi, prikažite na grafu, kjer vrednost na x osi ustreza prvemu latentnemu faktorju, vrednost na y osi pa drugemu latentnemu faktorju.

Oddaja: Na spletni učilnici oddajte poročilo in izvorno kodo (kot pri preteklih domačih nalogah). Poročilo mora biti napisano s predpisano predlogo. V poročilu na kratko opišite implementirane algoritme, prikažite rezultate in odgovorite na zgornja vprašanja.
