import math
from collections import defaultdict, Counter


def getEntropijo(razredi):
    skupno = len(razredi)
    stevci = Counter(razredi)
    entropija = 0.0
    for stevilo in stevci.values():
        p = stevilo / skupno
        if p > 0:
            entropija -= p * math.log2(p)
    return entropija


def tehtanaEntropijaSkupin(skupine, razredi, skupna_velikost):
    skupna_entropija = 0.0
    for indeksi in skupine.values():
        if not indeksi:
            continue
        velikost = len(indeksi)
        skupina_razredi = [razredi[i] for i in indeksi]
        ent = getEntropijo(skupina_razredi)
        skupna_entropija += (velikost / skupna_velikost) * ent
    return skupna_entropija


def grupirajPrimere(trenutne_skupine, znacilke, znacilka):
    nove_skupine = defaultdict(list)
    for kljuc, indeksi in trenutne_skupine.items():
        for i in indeksi:
            vrednost = znacilke[znacilka][i]
            nov_kljuc = kljuc + (vrednost,)
            nove_skupine[nov_kljuc].append(i)
    return nove_skupine


def dolociVecinskeRazredePoSkupinah(skupine, razredi):
    vecina_po_skupinah = {}
    for kljuc, indeksi in skupine.items():
        if indeksi:
            razredi_v_skupini = [razredi[i] for i in indeksi]
            vecina = Counter(razredi_v_skupini).most_common(1)[0][0]
            vecina_po_skupinah[kljuc] = vecina
    return vecina_po_skupinah


def getTocnost(znacilke, izbrane_znacilke, razredi, vecina_po_skupinah):
    pravilni = 0
    skupno = len(razredi)
    for i in range(skupno):
        kljuc = tuple(znacilke[znacilka][i] for znacilka in izbrane_znacilke)
        napoved = vecina_po_skupinah.get(kljuc)
        if napoved is None:
            napoved = Counter(razredi).most_common(1)[0][0]
        if napoved == razredi[i]:
            pravilni += 1
    return pravilni / skupno


def naloga2(znacilke: dict, razredi: list, koraki: int) -> tuple:
    stevilo_primerov = len(razredi)

    if koraki == 0:
        ent = getEntropijo(razredi)
        vecinski = Counter(razredi).most_common(1)[0][0]
        pravilni = sum(1 for r in razredi if r == vecinski)
        return ent, pravilni / stevilo_primerov

    vse_znacilke = list(znacilke.keys())
    izbrane_znacilke = []
    trenutne_skupine = defaultdict(list)
    trenutne_skupine[()] = list(range(stevilo_primerov))

    for _ in range(koraki):
        najboljsa_znacilka = None
        najmanjsa_entropija = float('inf')

        for kandidat in vse_znacilke:
            if kandidat in izbrane_znacilke:
                continue
            nove_skupine = grupirajPrimere(
                trenutne_skupine, znacilke, kandidat)
            ent = tehtanaEntropijaSkupin(
                nove_skupine, razredi, stevilo_primerov)
            if ent < najmanjsa_entropija:
                najmanjsa_entropija = ent
                najboljsa_znacilka = kandidat

        izbrane_znacilke.append(najboljsa_znacilka)
        trenutne_skupine = grupirajPrimere(
            trenutne_skupine, znacilke, najboljsa_znacilka)

    entropija = tehtanaEntropijaSkupin(
        trenutne_skupine, razredi, stevilo_primerov)
    vecina_po_skupinah = dolociVecinskeRazredePoSkupinah(
        trenutne_skupine, razredi)
    tocnost = getTocnost(
        znacilke, izbrane_znacilke, razredi, vecina_po_skupinah)

    return entropija, tocnost
