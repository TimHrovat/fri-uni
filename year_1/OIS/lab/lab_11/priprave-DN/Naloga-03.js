/**
 * Spodnjo vsebino skopirajte v začetni repozitorij DN
 * (https://github.com/OIS-2023-2024/DN) na konec datoteke streznik.js.
 *
 * Vprašanja:
 *  - Kakteri indeksi bodo v seznamu nedefinirani (angl. null) pri
 *    klicu storitve '/nalaganje-vsebine'?
 *    nal1, nal6 nista null
 *
 *  - Kakšno pa bo zaporedje izpisov v ukazni vrstici strežnika?
 *    nal1, nal6, nal4, nal2, nal3, nal5
 */

function cakamXSekund(x, povratniKlic) {
  const https = require("https");

  // Dodana izjema HTTPS, ki nimajo veljavnega certifikata
  const agent = new https.Agent({
    rejectUnauthorized: false,
  });

  // na odgovor storitve čakamo 3 sekunde
  axios
    .get("https://hub.dummyapis.com/delay?seconds=" + x, { httpsAgent: agent })
    .then(function (response) {
      // uspešni odgovor
      povratniKlic();
    })
    .catch(function (error) {
      // napaka
      console.log(error);
    });
}

streznik.get("/nalaganje-vsebine", (zahteva, odgovor) => {
  let nal1, nal2, nal3, nal4, nal5, nal6;
  console.log((nal1 = "Nalaganje strani..."));
  cakamXSekund(5, function () {
    console.log((nal2 = "podatki za prikaz grafa pripravljeni"));
    cakamXSekund(2, function () {
      console.log((nal3 = "podatki uporabnikov na voljo"));
    });
  });
  cakamXSekund(3, function () {
    console.log((nal4 = "podatki tabele na voljo"));
  });
  cakamXSekund(10, function () {
    console.log((nal5 = "AI modeli generirani"));
  });
  console.log((nal6 = "...priprava HTML"));
  odgovor.send({ naloge: [nal1, nal2, nal3, nal4, nal5, nal6] });
});

