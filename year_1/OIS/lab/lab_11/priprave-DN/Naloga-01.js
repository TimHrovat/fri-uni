/**
 * Spodnjo vsebino skopirajte v začetni repozitorij DN
 * (https://github.com/OIS-2023-2024/DN) na konec datoteke streznik.js.
 *
 * Vprašanja:
 *  - Kakšno vsebino tabele vam vrne klic storitve '/zaporedje' ?
 *    ["dež", "oblaki", "sonce"]
 *
 *  - Kaj se zgodi, če odkomentiramo komentar '// odgovor.send(tabela);',
 *    dobimo server error
 *
 *    ponovno zaženemo strežnik in pokličemo storitev '/zaporedje'?
 *      - Ali bo odgovor storitve v tem primeru seznam z dodatnim elementom 'megla'?
 *    ne
 */

streznik.get("/zaporedje", (zahteva, odgovor) => {
  let tabela = [];

  function funkcija1() {
    tabela.push("sonce");
    odgovor.send(tabela);
  }

  function funkcija2() {
    return "dež";
  }

  const https = require("https");

  // Dodana izjema HTTPS, ki nimajo veljavnega certifikata
  const agent = new https.Agent({
    rejectUnauthorized: false,
  });

  function funkcija3(povratniKlic) {
    // na odgovor storitve čakamo 5 sekund
    // ( Dokumentacija knjižnice axios: https://www.npmjs.com/package/axios )
    axios
      .get("https://hub.dummyapis.com/delay?seconds=5", { httpsAgent: agent })
      .then(function (response) {
        // uspešni odgovor
        povratniKlic("sneg");
      })
      .catch(function (error) {
        // napaka
        console.log(error);
      });
  }

  function funkcija4(povratniKlic) {
    tabela.push("oblaki");
    povratniKlic("megla");
  }

  tabela.push(funkcija2());
  funkcija4(function (odgovorFunkcije4) {
    funkcija3(function (odgovorFunkcije3) {
      tabela.push(odgovorFunkcije3);
    });
    funkcija1(odgovor);
    tabela.push(odgovorFunkcije4);
    //odgovor.send(tabela);
  });
});

