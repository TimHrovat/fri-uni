from collections import Counter


def naloga1(vhod: list, vhodS: list) -> tuple[list, list, float]:
    MAX_SIMBOLOV = 4096

    if not vhodS:  # Kodiranje
        slovar = {bytes([i]): i for i in range(256)}
        obratni_slovar = {i: bytes([i]) for i in range(256)}
        trenutni_indeks = 256
        izhod = [slovar[bytes([ord(b)])] for b in vhod]

        while trenutni_indeks < MAX_SIMBOLOV:
            pari = Counter(zip(izhod, izhod[1:]))

            if not pari:
                break

            najpogostejsi_par = max(pari, key=pari.get)

            if pari[najpogostejsi_par] < 2:
                break

            slovar[najpogostejsi_par] = trenutni_indeks
            obratni_slovar[trenutni_indeks] = obratni_slovar[najpogostejsi_par[0]] + obratni_slovar[najpogostejsi_par[1]]
            trenutni_indeks += 1

            nov_izhod = []
            i = 0
            while i < len(izhod):
                if i < len(izhod) - 1 and (izhod[i], izhod[i + 1]) == najpogostejsi_par:
                    nov_izhod.append(slovar[najpogostejsi_par])
                    i += 2
                else:
                    nov_izhod.append(izhod[i])
                    i += 1
            izhod = nov_izhod

        R = len(vhod) * 8 / len(izhod) / 12
        return izhod, list(slovar.items()), R
    else:  # Dekodiranje
        izhod = list(vhod)
        i = len(vhodS) - 1

        while i >= 256:
            for j in range(len(izhod)):
                if izhod[j] == i:
                    izhod[j] = vhodS[i][0]
                    izhod.insert(j+1, vhodS[i][1])
            i -= 1

        izhod = [chr(c) for c in izhod]
        R = (len(izhod) * 8) / (len(vhod) * 12)
        return izhod, [], R
