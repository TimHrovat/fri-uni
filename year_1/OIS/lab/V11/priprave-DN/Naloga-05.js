/**
 * V navodili DN-ja je bilo potrebno implementirati sledečo funkcionalnost:
 *  Na strani odjemalca je potrebno dodati poslušalca premik z miško na HTML element z enoličnim
 *  identifikatorjem podrobnostiIzleta. Ob premiku naj se izpiše najcenejša destinacija izbranega
 *  računa ter cena (npr. Najcenejša destinacija na računu je Antonijev Rov po ceni 16,35 €.). V
 *  primeru, da noben račun ni izbran naj se izpiše sporočilo `Noben račun ni izbran`.
 *
 * Izpolnite manjkakočo vsebino v nizih 'ODGOVOR', da bo funkcionalnost delovala.
 *
 * Pri reševanju si lahko pomagate tako, da skopirate spodnjo izvorno kodo v začetni repozitorij DN
 * (https://github.com/OIS-2023-2024/DN) v datoteko
 * 'public/skripte/prijava.js' in sicer v funkcijo, ki se proži, ko se spletno mesto naloži ('$(document).ready').
 */

document.getElementById("ODGOVOR").addEventListener("mouseover", (event) => {
  if (!event.target.attributes.racun)
    event.target.title = "Noben račun ni izbran.";
  else {
    let idIzbranegaRacuna = event.target.attributes.racun.value;

    $.get("/destinacije-racuna/" + idIzbranegaRacuna, (podatki) => {
      let cena = Number.MAX_SAFE_INTEGER;
      destinacijaIme = "";

      for (let i = 0; i < podatki.length; i++) {
        if (podatki[i].cena < cena) {
          cena = podatki[i].cena;
          destinacijaIme = podatki[i].ime;
        }
      }

      event.target.title =
        "Najcenejša destinacija na računu je " +
        destinacijaIme +
        " po ceni " +
        cena +
        " €.";
    });
  }
});

