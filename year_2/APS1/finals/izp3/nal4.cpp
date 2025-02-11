/**
(a) Časovna zahtevnost v odvisnosti od x, n, m:
        • Potrebujemo O(\log n) množenj.
        • Dolžina števil je omejena z m, zato je vsako množenje O(m \log m).
        • Skupna časovna zahtevnost: O(\log n \cdot m \log m).

(b) Časovna zahtevnost v odvisnosti od x in n (d(a) = \log_{10} a):
        • Dolžina števila med izračunom naraste približno linearno s številom množenj
          in je proporcionalna \log_{10} x^n = n \log_{10} x.
        • Za množenje dveh števil z dolžino d potrebujemo O(d \log d), kjer d \sim n \log_{10} x.
        • Skupna časovna zahtevnost: O(\log n \cdot n \log x \log(n \log x)).

(c) Krovni izrek (master theorem) in njegova omejitev:
Ta izrek velja za algoritme “deli in vladaj”, kjer se problem velikosti n deli
na a podproblemov, pri čemer je vsak podproblem velikosti n / b.

Pri potenciranju s kvadriranjem ne razdelimo problema na več podproblemov fiksnih
velikosti, ampak izkoriščamo lastnosti eksponenta n, da problem rekurzivno zmanjšamo
na polovico (n \to n / 2). Število podproblemov ni a > 1, ampak je vedno a = 1, kar pomeni,
da krovni izrek ne zajame tega primera. Poleg tega stroški združevanja (tj. množenja) niso
izraziti v obliki n^d, saj je vsak korak odvisen od dolžine števila, ki eksponentno narašča.
 */
