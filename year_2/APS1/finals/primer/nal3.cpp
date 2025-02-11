/**
 * Dijkstrov algoritem je namenjen iskanju najkrajših poti od začetnega vozlišča
 * do vseh vozlišč v povezanem, uteženem grafu (brez negativnih uteži).
 *
 * Glavna ideja:
 *  - Algoritem vzdržuje seznam najkrajših znanih razdalj do vsakega vozlišča
 *  - Začne pri začetnem vozlišču in razdaljo zanj nastavi na 0, za vsa ostala vozlišča pa neskončno
 *  - Iterativno obdeluje najbližje neobdelano vozlišče (tisto z najmanjšo trenutno razdalji),
 *    posodablja razdalje do njegovih sosedov, če najde krajšo pot
 *  - Ko so vsa vozlišča obdelana, algoritem vrne najkrajše poti
 *
 * 9. vrstica preveri, ali je trenutna razdalja (d) v prioritetni vrsti še vedno
 * enaka razdalji dist[x], ki je bila shranjena, ko je bilo vozlišče x nazadnje
 * obdelano. Če ni enaka, pomeni, da je bilo vozlišče že obdelano z manjšo razdaljo,
 * zato to iteracijo preskočimo
 *
 * Pravilnost:
 *  - Algoritem bi še vedno deloval pravilno, saj se nove, boljše poti vedno
 *    posodabljajo z najmanjšimi razdaljami v prioritetni vrsti
 *
 * Časovna zahtevnost:
 *  - Brez te vrstice se časovna zahtevnost poveča saj nekatera vozlišča obdelujemo večkrat
 */


