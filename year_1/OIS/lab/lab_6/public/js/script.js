const premakniPesemIzSeznamaVKosarico = (id, naziv, izvajalec, cena, stopnja, azuriraj) => {
    if (azuriraj) $.get("/kosarica/" + id, (podatki) => {
        /* Dodaj izbramo pesem v sejo */
    });

    let ddv = "";
    // odkomentiraj za reševanje 2. naloge
    ddv = "(<span class='stopnja'>" + stopnja + "</span> %)";

    // Dodaj pesem v desni seznam
    $("#kosarica").append("<div id='" + id + "' class='pesem'> \
        <button type='button' class='btn btn-light btn-sm'> \
          <i class='fa-solid fa-minus'></i> \
          <strong><span class='naziv'>" + naziv + "</span></strong> \
          (<span class='izvajalec'>" + izvajalec + "</span>) \
          @ <span class='cena'>" + cena + "</span> € \
           " + ddv + " \
        </button> \
      </div>");

    // Dogodek ob kliku na pesem v košarici (na desnem seznamu)
    $("#kosarica #" + id).click(function () {
        let pesem_kosarica = $(this);
        $.get("/kosarica/" + id, (podatki) => {
            /* Odstrani izbrano pesem iz seje */
            // Če je košarica prazna, onemogoči gumbe za pripravo računa
            if (!podatki || podatki.length == 0) {
                $("#racun_html").prop("disabled", true);
                $("#racun_xml").prop("disabled", true);
            }
        });
        // Izbriši pesem iz desnega seznama
        pesem_kosarica.remove();
        // Pokaži pesem v levem seznamu
        $("#pesmi #" + id).show();
    });

    // Skrij pesem v levem seznamu
    $("#pesmi #" + id).hide();
    // Ker košarica ni prazna, omogoči gumbe za pripravo računa
    $("#racun_html").prop("disabled", false);
    $("#racun_xml").prop("disabled", false);
};

$(document).ready(() => {
    // Posodobi podatke iz košarice na spletni strani
    $.get("/kosarica", (kosarica) => {
        kosarica.forEach((pesem) => {
            let stevilo = 1;
            // odkomentiraj za reševanje 2. naloge
            stevilo = pesem.stevilo;
            premakniPesemIzSeznamaVKosarico(pesem.stevilkaArtikla,
                                            pesem.opisArtikla.split(" (")[0],
                                            pesem.opisArtikla.split(" (")[1].split(")")[0],
                                            (pesem.cena * (1 + pesem.stopnja / stevilo)).toFixed(2),
                                            pesem.stopnja,
                                            false);
        });
    });

    // Klik na pesem v levem seznamu sproži
    // dodajanje pesmi v desni seznam (košarica)
    $("#pesmi .pesem").click(function () {
        let pesem = $(this);

        premakniPesemIzSeznamaVKosarico(pesem.attr("id"),
                                        pesem.find("button .naziv").text(),
                                        pesem.find("button .izvajalec").text(),
                                        pesem.find("button .cena").text(),
                                        pesem.find("button .stopnja").text(),
                                        true);
    });

    // Klik na gumba za pripravo računov
    $("#racun_html").click(() => (window.location = "/izpisiRacun/html"));
    $("#racun_xml").click(() => (window.location = "/izpisiRacun/xml"));
});
