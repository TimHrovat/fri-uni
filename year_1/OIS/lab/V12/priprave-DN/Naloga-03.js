/**
 * V navodili DN-ja je bilo potrebno implementirati sledečo funkcionalnost,
 * ki je del funkcionalnosti iskanje denarnice:
 *
 * 2. Na strani odjemalca v datoteki public/skripte/seznam.js najprej omogočite izpis
 * oddaljenosti na legendo na desnem zgornjem kotu v zemljevidu pod sporočilom Kje je
 * izgubljena denarnica? in sicer tako, da izračunate oddaljenost s pomočjo funkcije v
 * JavaScript knjižnici public/skripte/oddaljenost.js. Oddaljenost naj se neposredno
 * prikazuje, ko se z miško premikate po zemljevidu. Izpis najdene lokacije v legendo
 * zemljevida naj se prikaže, ko je oddaljenost med destinacijo izgubljene denarnico
 * ter miško manjša od 20 km.
 *
 * 3. Da se uporabniku poenostavi iskanje, mu je treba podati namig s spreminjanjem barve
 *  roba zemljevida, kjer rdeča barva predstavlja bližino, modra barva pa oddaljenost,
 *  podobno kot po principu otroške igre iskanja predmetov z namigom “hladno-vroče”. V
 *  primeru, da je oddaljenost miške od izgubljene denarnice večja od 255 pikslov je
 *  barva pisave povsem modra (npr. RGB(0,0,255)), ko pa je oddaljenost manjša od 255
 *  pikslov intenziteta barve prehaja proti rdeči barvi (npr. pri oddaljenosti 80px je
 *  RGB predstavitev (175, 0, 80) kjer je modra barva 255 - rdeča barva.
 *
 * Izpolnite manjkakočo vsebino v nizih 'ODGOVOR', da bo funkcionalnost delovala.
 *
 * Pri reševanju si lahko pomagate tako, da skopirate spodnjo izvorno kodo v začetni repozitorij DN
 * (https://github.com/OIS-2023-2024/DN) v datoteko
 * 'public/skripte/seznam.js' in sicer v funkcijo, ki se proži, ko se spletno mesto naloži ('$(document).ready').
 */

/**
 *  funkcija oddaljenost
 * @param x1, x koordinata prve točke
 * @param y1, y koordinata prve točke
 * @param x2, x koordinata druge točke
 * @param y2, y koordinata druge točke
 * @return {oddaljenost v pikslih * 100}
 */
function oddaljenostTock(x1, y1, x2, y2) {
  var aa = x1 - x2;
  var bb = y1 - y2;

  return Math.sqrt(aa * aa + bb * bb) * 100;
}

let nasel = false;

mapa.on("mousemove", (event) => {
  console.log(event.latlng.lat + " | " + event.latlng.lng);

  var oddaljenost = oddaljenostTock(
    document.getElementById("idDestinacijeDenarnice").getAttribute("sirina"),
    document.getElementById("idDestinacijeDenarnice").getAttribute("dolzina"),
    event.latlng.lat,
    event.latlng.lng,
  );

  var rdeca;
  var modra;
  if (oddaljenost > 255) {
    modra = 255;
    rdeca = 0;
  } else {
    rdeca = 255 - oddaljenost;
    modra = 255 - rdeca;
  }

  document.getElementById("mapa_id").style.borderColor =
    "rgb(" + rdeca + ",0," + modra + ")";

  if (oddaljenost < 20 && !nasel) {
    nasel = true;
    premesajImeDestinacijeInPrikaziNaLegendi(info);
  } else if (!nasel) info.update(oddaljenost);
});

