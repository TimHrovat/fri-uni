if (!process.env.PORT) process.env.PORT = 8080;

const axios = require('axios');

// Priprava povezave na podatkovno bazo
const sqlite3 = require("sqlite3").verbose();
const pb = new sqlite3.Database("DestinationInvoice.sl3");

// Priprava dodatnih knjižnic
const formidable = require("formidable");

// Priprava strežnika
const express = require("express");
const streznik = express();
streznik.set("view engine", "hbs");
streznik.use(express.static("public"));

// Podpora sejam na strežniku
const expressSession = require("express-session");

// Globalne spremenljivke strežnika
const VELIKOST_SLOVENIJE_KM = 20273.0;
let destinacijaIzgubljeneDenarnice;

streznik.use(expressSession({
    secret: "123456789QWERTY", // Skrivni ključ za podpisovanje piškotov
    saveUninitialized: true, // Novo sejo shranimo
    resave: false, // Ne zahtevamo ponovnega shranjevanja
    cookie: {
        maxAge: 3600000, // Seja poteče po 1 h neaktivnosti
    },
}));

// Vrne naziv stranke (ime in priimek) glede na ID stranke
const vrniNazivStranke = (strankaId, povratniKlic) => {
    pb.all("SELECT Customer.FirstName || ' ' || Customer.LastName AS naziv \
         FROM   Customer \
         WHERE  Customer.CustomerId = $id", {$id: strankaId}, (napaka, vrstica) => {
        if (napaka) povratniKlic(""); else povratniKlic(vrstica.length > 0 ? vrstica[0].naziv : "");
    });
};

// Vrne državo stranke glede na ID stranke
const vrniDrzavoStranke = (strankaId, povratniKlic) => {
    pb.all("SELECT Customer.Country AS drzava \
         FROM   Customer \
         WHERE  Customer.CustomerId = $id", {$id: strankaId}, (napaka, vrstica) => {
        if (napaka) povratniKlic(""); else povratniKlic(vrstica.length > 0 ? vrstica[0].drzava : "");
    });
};

// Vrni seznam destinacij (atrakcije, hostli, živalski vrtovi)
const vrniSeznamDestinacij = (povratniKlic) => {
    pb.all("SELECT   Destination.OsmId AS id, \
                  Destination.Name AS ime, \
                  Destination.NameEN AS imeAnglesko, \
                  Destination.Wikidata AS wikidata, \
                  Destination.Website AS website, \
                  Destination.City AS mesto, \
                  DestinationType.Name AS vrstaDestinacije, \
                  COUNT(InvoiceLine.InvoiceId) AS steviloProdaj, \
                  Geometry.Type AS oblikaObmocja, \
                  Coordinate.lat AS zemljepisnaSirina, \
                  Coordinate.lng AS zemljepisnaDolzina \
         FROM     Destination, DestinationType, Geometry, Coordinate, InvoiceLine \
         WHERE    Destination.GeometryId = Geometry.GeometryId AND \
                  Destination.OsmId = InvoiceLine.DestinationId AND \
                  Destination.DestinationTypeId = DestinationType.DestinationTypeId AND \
                  Geometry.GeometryId = Destination.GeometryId AND \
                  Geometry.GeometryId = Coordinate.GeometryId \
         GROUP BY Destination.OsmId \
         ORDER BY ime ASC, vrstaDestinacije DESC \
         LIMIT    1000", (napaka, vrstice) => {
        povratniKlic(napaka, vrstice);
    });
};

// Vrni podrobnosti destinacije v košarici iz podatkovne baze
const destinacijeIzKosarice = (zahteva, povratniKlic) => {
    // Če je košarica prazna
    if (!zahteva.session.kosarica || zahteva.session.kosarica.length == 0) {
        povratniKlic([]);
    } else {
        pb.all("SELECT   Destination.OsmId AS id, \
                  Destination.Name AS ime, \
                  Destination.NameEN AS imeAnglesko, \
                  Destination.Wikidata AS wikidata, \
                  Destination.Website AS website, \
                  Destination.City AS mesto, \
                  DestinationType.Name AS vrstaDestinacije, \
                  Geometry.Type AS oblikaObmocja, \
                  Coordinate.lat AS zemljepisnaSirina, \
                  Coordinate.lng AS zemljepisnaDolzina \
         FROM     Destination, DestinationType, Geometry, Coordinate, InvoiceLine \
         WHERE    Destination.GeometryId = Geometry.GeometryId AND \
                  Destination.OsmId = InvoiceLine.DestinationId AND \
                  Destination.DestinationTypeId = DestinationType.DestinationTypeId AND \
                  Geometry.GeometryId = Destination.GeometryId AND \
                  Geometry.GeometryId = Coordinate.GeometryId AND \
                  Destination.OsmId IN (" + zahteva.session.kosarica.join(",") + ") \
         GROUP BY Destination.OsmId", (napaka, vrstice) => {
            povratniKlic(napaka, vrstice);
        });
    }
};

// Vrni podrobnosti destinacij na računu
const destinacijeIzRacuna = function (racunId, povratniKlic) {
    pb.all("SELECT DISTINCT Destination.OsmId AS osmId, \
                Destination.Name AS ime, \
                Destination.NameEN AS imeAnglesko, \
                Destination.Wikidata AS wikidata, \
                Destination.Website AS website, \
                DestinationType.Name AS vrstaDestinacije, \
                InvoiceLine.UnitPrice AS cena, \
                1 AS kolicina, \
                0 AS popust \
         FROM   Destination, DestinationType, Invoice, InvoiceLine \
         WHERE  DestinationType.DestinationTypeId = Destination.DestinationTypeId AND \
                Destination.OsmId = InvoiceLine.DestinationId AND \
                Destination.OsmId IN ( \
                  SELECT InvoiceLine.DestinationId \
                  FROM   InvoiceLine, Invoice \
                  WHERE  InvoiceLine.InvoiceId = Invoice.InvoiceId AND \
                         Invoice.InvoiceId = $id \
                ) GROUP BY Destination.OsmId ", {$id: racunId}, (napaka, vrstice) => {
        if (napaka)
            povratniKlic(false);
        else
            povratniKlic(napaka, vrstice);
    });
};

// Vrni lokacijo destinacije
const lokacijaDestinacije = function (destinacijaId, povratniKlic) {
    pb.get("SELECT DISTINCT Coordinate.lat AS zemljepisnaSirina, \
                  Coordinate.lng AS zemljepisnaDolzina \
         FROM     Destination, Geometry, Coordinate \
         WHERE    Destination.GeometryId = Geometry.GeometryId AND \
                  Geometry.GeometryId = Coordinate.GeometryId AND \
                  Destination.OsmId = $id ",
        {$id: destinacijaId}, (napaka, vrstice) => {
            if (napaka)
                povratniKlic(false);
            else
                povratniKlic(napaka, vrstice);
        });
};


// Vrni podrobnosti o stranki iz računa
const strankaIzRacuna = (racunId, povratniKlic) => {
    pb.all("SELECT Customer.* \
         FROM   Customer, Invoice \
         WHERE  Customer.CustomerId = Invoice.CustomerId AND \
                Invoice.InvoiceId = $id", {$id: racunId}, function (napaka, vrstice) {
        if (napaka) povratniKlic(false); else povratniKlic(vrstice[0]);
    });
};

// Vrni podrobnosti o stranki iz seje
const strankaIzSeje = (zahteva, povratniKlic) => {
    if (!zahteva.session.trenutnaStranka) {
        povratniKlic(false);

    } else {
        pb.all("SELECT * \
             FROM   Customer \
             WHERE  Customer.CustomerId = $id", {$id: zahteva.session.trenutnaStranka}, (napaka, vrstice) => {
            if (napaka) povratniKlic(false); else povratniKlic(vrstice[0]);
        });
    }
};

// Vrni stranke iz podatkovne baze
const vrniStranke = (povratniKlic) => {
    pb.all("SELECT * FROM Customer", (napaka, stranke) => {
        povratniKlic(napaka, stranke);
    });
};

// Vrni podrobnosti stranke iz podatkovne baze
const vrniPodrobnostiStranke = (povratniKlic) => {
    pb.one("SELECT * FROM Customer WHERE  Customer.CustomerId = ", (napaka, stranke) => {
        povratniKlic(napaka, stranke);
    });
};

// Vrni račune iz podatkovne baze
const vrniRacune = (povratniKlic) => {
    pb.all("SELECT Customer.FirstName || ' ' || Customer.LastName || \
                  ' (' || Invoice.InvoiceId || ') - ' || \
                  DATE(Invoice.InvoiceDate) AS Naziv, \
                Customer.CustomerId AS IdStranke, \
                Invoice.InvoiceId \
         FROM   Customer, Invoice \
         WHERE  Customer.CustomerId = Invoice.CustomerId", (napaka, vrstice) => povratniKlic(napaka, vrstice));
};

/**
 * Returns a random integer between min (inclusive) and max (inclusive).
 * The value is no lower than min (or the next integer greater than min
 * if min isn't an integer) and no greater than max (or the next integer
 * lower than max if max isn't an integer).
 * Using Math.round() will give you a non-uniform distribution!
 */
function getRandomInt(min, max) {
    min = Math.ceil(min);
    max = Math.floor(max);
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

function generirajDestinacijoDenarnice(destinacije, povratniKlic) {
    destinacijaIzgubljeneDenarnice = destinacije[getRandomInt(0, destinacije.length - 1)];
    povratniKlic(destinacijaIzgubljeneDenarnice);
}

// Prikaz začetne strani
streznik.get("/", (zahteva, odgovor) => {
    // v primeru, da uporabnik ni prijavljen
    if (!zahteva.session.trenutnaStranka) {
        odgovor.redirect("/prijava");
    } else {
        vrniSeznamDestinacij((napaka, destinacije) => {
            vrniNazivStranke(zahteva.session.trenutnaStranka, (nazivOdgovor) => {
                generirajDestinacijoDenarnice(destinacije, (destinacijaDenarnice) => {
                    odgovor.render("seznam", {
                        podnaslov: "Nakupovalni seznam",
                        prijavniGumb: "Odjava",
                        seznamDestinacij: destinacije,
                        izgubljenaDenarnica: destinacijaDenarnice,
                        nazivStranke: zahteva.session.trenutnaStranka ? nazivOdgovor : "gosta",
                    });
                });
            });
        });
    }
});

// Dodajanje oz. brisanje destinacij iz košarice
streznik.get("/kosarica/:idOsm", (zahteva, odgovor) => {
    let idOsm = parseInt(zahteva.params.idOsm, 10);

    if (!zahteva.session.kosarica) zahteva.session.kosarica = [];

    if (zahteva.session.kosarica.indexOf(idOsm) > -1) {
        // Če je destinacija v košarici, jo izbrišemo
        zahteva.session.kosarica.splice(zahteva.session.kosarica.indexOf(idOsm), 1);
    } else {
        // Če destinacije ni v košarici, jo dodamo
        zahteva.session.kosarica.push(idOsm);
    }
    // V odgovoru vrnemo vsebino celotne košarice
    odgovor.send(zahteva.session.kosarica);
});

// Vrni podrobnosti košarice
streznik.get("/kosarica", (zahteva, odgovor) => {
    destinacijeIzKosarice(zahteva, (napaka, destinacije) => {
        if (napaka || !destinacije) odgovor.sendStatus(500); else odgovor.send(destinacije);
    });
});

// Vrni podrobnosti o destinaciji
streznik.get("/vec-o-destinaciji-api/:id", (zahteva, odgovor) => {
    let idDestinacije = zahteva.params.id;
    lokacijaDestinacije(idDestinacije, (napaka, geometrijaDestinacije) => {
        odgovor.send("");
    });
});

// Vrni podronosti o stranki in sicer velikost države stranke v primerjavi
// s Slovenijo ter prisotni jeziki
streznik.get("/vec-o-stranki-api/:id", (zahteva, odgovor) => {
    let idStranke = zahteva.params.id;
    odgovor.send("");
});

// Izpis račun v HTML predstavitvi na podlagi podatkov iz baze
streznik.post("/izpisiRacunBaza", (zahteva, odgovor) => {
    let form = new formidable.IncomingForm();
    form.parse(zahteva, (napaka, polja) => {
        let racunId = parseInt(polja["seznamRacunov"], 10);
        strankaIzRacuna(racunId, (stranka) => {
            destinacijeIzRacuna(racunId, (napaka, destinacije) => {
                odgovor.setHeader("Content-Type", "text/xml");

                let povzetek = {
                    vsotaSPopustiInDavki: 0,
                    vsoteZneskovDdv: {0: 0, 9.5: 0, 22: 0},
                    vsoteOsnovZaDdv: {0: 0, 9.5: 0, 22: 0},
                    vsotaVrednosti: 0,
                    vsotaPopustov: 0,
                };

                destinacije.forEach((destinacija, i) => {
                    destinacija.zapSt = i + 1;
                    destinacija.vrednost = destinacija.kolicina * destinacija.cena;
                    destinacija.davcnaStopnja = 22;

                    destinacija.popustStopnja = 0;
                    destinacija.popust = destinacija.kolicina * destinacija.cena * (destinacija.popustStopnja / 100);

                    destinacija.osnovaZaDdv = destinacija.vrednost - destinacija.popust;
                    destinacija.ddv = destinacija.osnovaZaDdv * (destinacija.davcnaStopnja / 100);
                    destinacija.osnovaZaDdvInDdv = destinacija.osnovaZaDdv + destinacija.ddv;

                    povzetek.vsotaSPopustiInDavki += destinacija.osnovaZaDdv + destinacija.ddv;
                    povzetek.vsoteZneskovDdv["" + destinacija.davcnaStopnja] += destinacija.ddv;
                    povzetek.vsoteOsnovZaDdv["" + destinacija.davcnaStopnja] += destinacija.osnovaZaDdv;
                    povzetek.vsotaVrednosti += destinacija.vrednost;
                    povzetek.vsotaPopustov += destinacija.popust;
                });
                odgovor.render("eslog", {
                    vizualiziraj: true,
                    postavkeRacuna: destinacije,
                    povzetekRacuna: povzetek,
                    stranka: stranka,
                    layout: null,
                });
            });
        });
    });
});

// Izpis računa v HTML predstavitvi ali izvorni XML obliki
streznik.get("/izpisiRacun/:oblika", (zahteva, odgovor) => {
    strankaIzSeje(zahteva, (stranka) => {
        destinacijeIzKosarice(zahteva, (napaka, destinacije) => {
            if (napaka || !destinacije) {
                odgovor.sendStatus(500);
            } else if (destinacije.length == 0) {
                odgovor.send("<p>V košarici nimate nobene destinacije, " + "zato računa ni mogoče pripraviti!</p>");
            } else {
                let povzetek = {
                    vsotaSPopustiInDavki: 0,
                    vsoteZneskovDdv: {0: 0, 9.5: 0, 22: 0},
                    vsoteOsnovZaDdv: {0: 0, 9.5: 0, 22: 0},
                    vsotaVrednosti: 0,
                    vsotaPopustov: 0,
                };

                destinacije.forEach((destinacija, i) => {
                    destinacija.zapSt = i + 1;
                    destinacija.cena = destinacija.vrstaDestinacije == "živalski vrt" ? 8.5 :
                        (destinacija.vrstaDestinacije == "atrakcija" ? 17.5 : 12.0);
                    destinacija.kolicina = 1;
                    destinacija.vrednost = destinacija.kolicina * destinacija.cena;
                    destinacija.davcnaStopnja = 22;

                    destinacija.popustStopnja = 0;

                    destinacija.popust = destinacija.kolicina * destinacija.cena * (destinacija.popustStopnja / 100);

                    destinacija.osnovaZaDdv = destinacija.vrednost - destinacija.popust;
                    destinacija.ddv = destinacija.osnovaZaDdv * (destinacija.davcnaStopnja / 100);
                    destinacija.osnovaZaDdvInDdv = destinacija.osnovaZaDdv + destinacija.ddv;

                    povzetek.vsotaSPopustiInDavki += destinacija.osnovaZaDdv + destinacija.ddv;
                    povzetek.vsoteZneskovDdv["" + destinacija.davcnaStopnja] += destinacija.ddv;
                    povzetek.vsoteOsnovZaDdv["" + destinacija.davcnaStopnja] += destinacija.osnovaZaDdv;
                    povzetek.vsotaVrednosti += destinacija.vrednost;
                    povzetek.vsotaPopustov += destinacija.popust;
                });

                odgovor.setHeader("Content-Type", "text/xml");
                odgovor.render("eslog", {
                    vizualiziraj: zahteva.params.oblika == "html",
                    postavkeRacuna: destinacije,
                    povzetekRacuna: povzetek,
                    stranka: stranka,
                    layout: null,
                });
            }
        });
    });
});

// Privzeto izpiši račun v HTML obliki
streznik.get("/izpisiRacun", (zahteva, odgovor) => {
    odgovor.redirect("/izpisiRacun/html");
});

function pridobiStatistikoStrank(stranke) {
    let drzave = {};
    let telefon = 0;
    let faks = 0;
    let email = 0;
    let zaposleni = 0;
    stranke.forEach((stranka, i) => {
        if (drzave[stranka.Country])
            drzave[stranka.Country]++;
        else
            drzave[stranka.Country] = 1;

        if (stranka.Phone) telefon++;
        if (stranka.Fax) faks++;
        if (stranka.Email) email++;
        if (stranka.Company) zaposleni++;

    });
    return {
        vsi: stranke.length,
        razlicne_drzave: Object.keys(drzave).length,
        zaposlenih: zaposleni,
        komunikacija: {
            email: email,
            neuporabljeno: [{telefon: telefon}, {golobi: 0}, {mahanje: 1}, {faks: faks}, {kričanje: -1}]
        }
    };
}

// Prikaz strani za prijavo
streznik.get("/prijava", (zahteva, odgovor) => {
    vrniStranke((napaka1, stranke) => {
        let statistikaStrank = pridobiStatistikoStrank(stranke);

        vrniRacune((napaka2, racuni) => {
            for (let i = 0; i < stranke.length; i++) stranke[i].stRacunov = 0;

            odgovor.render("prijava", {
                sporocilo: "",
                prijavniGumb: "Prijava uporabnika",
                podnaslov: "Prijavna stran",
                seznamStrank: stranke,
                seznamRacunov: racuni,
                statistika: statistikaStrank
            });
        });
    });
});

// Prijava ali odjava stranke
streznik.get("/prijavaOdjava/:strankaId", (zahteva, odgovor) => {
    if (zahteva.get("referer").endsWith("/prijava")) {
        // Izbira stranke oz. prijava
        zahteva.session.trenutnaStranka = parseInt(zahteva.params.strankaId, 10);
        odgovor.redirect("/");
    } else {
        // Odjava stranke
        delete zahteva.session.trenutnaStranka;
        delete zahteva.session.kosarica;
        odgovor.redirect("/prijava");
    }
});

// Prikaz seznama destinacij na strani
streznik.get("/podroben-seznam-destinacij", (zahteva, odgovor) => {
    vrniSeznamDestinacij((napaka, vrstice) => {
        if (napaka) odgovor.sendStatus(500); else odgovor.send(vrstice);
    });
});

streznik.get("/destinacije-racuna/:racunId", (zahteva, odgovor) => {
    let racunId = parseInt(zahteva.params.racunId, 10);
    destinacijeIzRacuna(racunId, (napaka, vrstice) => {
        odgovor.send(vrstice);
    });
});

streznik.get("/opis", (zahteva, odgovor) => {
    odgovor.render("opis", {
        prijavniGumb: "Prijava uporabnika", podnaslov: "Opis",
    });
});

streznik.get("/nacrt", (zahteva, odgovor) => {
    odgovor.render("nacrt", {
        prijavniGumb: "Prijava uporabnika", podnaslov: "Načrt",
    });
});

streznik.get("/preveri/:id", (zahteva, odgovor) => {
    if (destinacijaIzgubljeneDenarnice.id == zahteva.params.id) {
        odgovor.send("Nedokončana funkcionalnost iskanja izgubljene denarnice.");
    } else
        odgovor.send("Tukaj ni izgubljene denarnice Vitalika!");
});


streznik.listen(process.env.PORT, () => {
    console.log(`Strežnik je pognan na vratih ${process.env.PORT}!`);
});
