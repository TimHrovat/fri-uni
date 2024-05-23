/**
 * Spodnjo vsebino skopirajte v začetni repozitorij DN
 * (https://github.com/OIS-2023-2024/DN) na konec datoteke streznik.js.
 *
 * Vprašanja:
 *  - Kakšna vsebino niza vam vrne klic storitve '/novo-zaporedje' ?
 *  1534627
 */

streznik.get("/novo-zaporedje", (zahteva, odgovor) => {
  let niz = "";

  function funkcija1() {
    niz += "3";
  }

  function funkcija2() {
    niz += "5";
    return "2";
  }

  function funkcija3(povratniKlic) {
    niz += "1";
    povratniKlic("6");
  }

  function funkcija4(povratniKlic) {
    niz += "4";
    povratniKlic("7");
  }

  // odgovorFunkcije3 = "6"
  funkcija3(function (odgovorFunkcije3) {
    let rezultatFunkcije2 = funkcija2();
    funkcija1();

    funkcija4(function (odgovorFunkcije4) {
      niz += odgovorFunkcije3;
      niz += rezultatFunkcije2;
      niz += odgovorFunkcije4;
      odgovor.send(niz);
    });
  });
});

