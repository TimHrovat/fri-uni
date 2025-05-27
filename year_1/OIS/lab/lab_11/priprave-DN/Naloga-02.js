/**
 * Spodnjo vsebino skopirajte v začetni repozitorij DN
 * (https://github.com/OIS-2023-2024/DN) na konec datoteke streznik.js.
 *
 * Vprašanja:
 *  - Kakšna bo vrednost ključa 'rezultat' pri klicu storitve '/izracun'?
 *  vrednost bo 1
 */

streznik.get("/izracun", (zahteva, odgovor) => {
  let rezultat = 0;

  function funkcija1() {
    rezultat -= 5;
  }

  function funkcija2() {
    return 3;
  }

  const https = require("https");

  // Dodana izjema HTTPS, ki nimajo veljavnega certifikata
  const agent = new https.Agent({
    rejectUnauthorized: false,
  });

  function funkcija3(povratniKlic) {
    // na odgovor storitve čakamo 5 sekund
    axios
      .get("https://hub.dummyapis.com/delay?seconds=5", { httpsAgent: agent })
      .then(function (response) {
        // uspešni odgovor
        povratniKlic(-7);
      })
      .catch(function (error) {
        // napaka
        console.log(error);
      });
  }

  function funkcija4(povratniKlic) {
    povratniKlic(1);
  }

  rezultat += funkcija2(); // 0 + 3 = 3
  // odgovorFunkcije4 ... 1
  // odgovorFunkcije3 ... -7
  funkcija4(function (odgovorFunkcije4) {
    funkcija3(function (odgovorFunkcije3) {
      rezultat += odgovorFunkcije3;
    });
    rezultat += odgovorFunkcije4; // 3 + 1 = 4
    funkcija1(); // rezultat = -1
    odgovor.send({ rezultat: -1 * rezultat });
    funkcija1();
  });
});

