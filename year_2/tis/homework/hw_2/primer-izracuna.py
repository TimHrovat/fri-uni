# Uvozimo vse potrebne module
import json
from math import log2
from collections import Counter

# Nalozimo podatke (datoteka 1.json mora biti v isti mapi kot ta skripta)
# Pri domaci nalogi vam tega koraka ni treba narediti, saj to za vas naredi skripta test_naloga2.py
file = open('./primeri/1.json')
datajson = json.load(file)
file.close()
A = datajson["podatki"]["znacilke"]
R = list(datajson["podatki"]["razredi"].values())[0]
print(A)
print(R)

# Naredimo postopek za znacilnico "Nebo" za prvi nivo odlocitvenega drevesa
# Ustvarimo seznam unikatnih vrednosti zancilnice "Nebo"
nodes = list(set(A["Nebo"]))

# V terke zlepimo vrednost znacilnice in pripadajoco vrednost razreda
AR = list(zip(A["Nebo"], R))

# Prestejemo vse zapise
N = len(AR)

# Naredimo seznam zapisov, ki pripadajo vsakemu od listov (lambda je tu zaradi enovrsticnice, sicer bi lahko napisali svojo funkcijo)
# Spodnja koda predpostavlja, da je na prvem mestu v seznamu "nodes" vrednost 'j' in na drugem mestu vrednost 'o'
list_j = list(filter(lambda z: z[0] == nodes[0], AR))
list_j = list(map(lambda z: z[1], list_j))
list_o = list(filter(lambda z: z[0] == nodes[1], AR))
list_o = list(map(lambda z: z[1], list_j))

# Enako kot zgoraj lahko nardimo tudi s sintakso za "list comprehension"
list_j = [z[1] for z in AR if z[0] == nodes[0]]
list_o = [z[1] for z in AR if z[0] == nodes[1]]

# Prestejemo zapise v obeh listih za vsako vrednost razreda
nj = len(list_j)
pj = Counter(list_j)

no = len(list_o)
po = Counter(list_o)

# Izracunamo entropijo za vsak list
Hj = sum(-(pj[i] / nj)*log2(pj[i]/nj) for i in pj)
Ho = sum(-(po[i] / no)*log2(po[i]/no) for i in po)

# Izracunamo utezeno vsoto entropij
H = nj/N*Hj+no/N*Ho
print(H)

# Izracunajmo se tocnost
# Poiscimo vecinski razred za vsak list
vecinski_o = po.most_common(1)
vecinski_j = pj.most_common(1)
# Izracun tocnosi
T = (vecinski_o[0][1] + vecinski_j[0][1])/N
print(T)

# Namig: Za kombinacijo znacilnic je postopek prakticno enak, le da najprej zdruzimo zelene znacilnice
# Primer za znacilnici "Nebo" in "Veter":
A2 = list(zip(A["Nebo"], A["Veter"]))
AR = list(zip(A2, R))
