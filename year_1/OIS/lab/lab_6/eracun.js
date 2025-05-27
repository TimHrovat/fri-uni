if (!process.env.PORT) process.env.PORT = 3000;

// Priprava povezave na podatkovno bazo
const sqlite3 = require("sqlite3").verbose();
const pb = new sqlite3.Database("Chinook.sl3");

// Priprava strežnika
const express = require("express");
const streznik = express();
streznik.set("view engine", "hbs");
streznik.use(express.static("public"));

// Podpora sejam na strežniku
const expressSession = require("express-session");
streznik.use(
    expressSession({
        // Skrivni ključ za podpisovanje piškotov
        secret: "123456789QWERTY",
        // Novo sejo shranimo
        saveUninitialized: true,
        // Ne zahtevamo ponovnega shranjevanja
        resave: false,
        cookie: {
            // Seja poteče po 1 h neaktivnosti
            maxAge: 3600000,
        },
    })
);

const razmerje_USD_EUR = 0.84;

// Izračun davčne stopnje glede na izvajalca in žanr
const davcnaStopnja = (izvajalec, zanr) => {
    switch (izvajalec) {
        case "Queen":
        case "Led Zappelin":
        case "Kiss":
            return 0;
        case "Justin Bieber":
        case "Incognito":
            return 22;
        default:
            break;
    }
    switch (zanr) {
        case "Metal":
        case "Heavy Metal":
        case "Easy Listening":
            return 0;
        default:
            return 9.5;
    }
};

// Določitev barve glede na naziv žanra
const pridobiBarvoZanra = (zanr) => {
    switch (zanr) {
        case "Reggae":
            return "#00FF00";
        case "Alternative & Punk":
            return "#FF4500";
        case "Latin":
            return "#FFD700";
        case "R&B/Soul":
            return "#7B68EE";
        case "Metal":
            return "#A9A9A9";
        case "Rock":
            return "#FF0000";
        case "Pop":
            return "#FFC0CB";
        case "Jazz":
            return "#1E90FF";
        case "TV Shows":
            return "#FFF8DC";
        case "Drama":
            return "#800080";
        case "Blues":
            return "#0000FF";
        default:
            return "#FFFFFF";
    }
};

// Prikaz seznama pesmi na strani
streznik.get("/", (zahteva, odgovor) => {
    // odkomentiraj za reševanje 2. naloge
    const tabela = ", Genre";
    const podatek = ", Genre.Name AS zanr ";
    const pogoj = "AND Track.GenreId = Genre.GenreId";

    pb.all(
        "SELECT   Track.TrackId AS id, \
                  TRACK.Name AS pesem, \
                  Artist.Name AS izvajalec, \
                  ROUND(Track.UnitPrice * " +
        razmerje_USD_EUR +
        ", 2) AS cena, \
                COUNT(InvoiceLine.InvoiceId) AS steviloProdaj\
                " + podatek + " \
       FROM     Track, Album, Artist, InvoiceLine " + tabela + " \
       WHERE    Track.AlbumId = Album.AlbumId AND \
                Artist.ArtistId = Album.ArtistId AND \
                InvoiceLine.TrackId = Track.TrackId \
                " + pogoj + " \
       GROUP BY Track.TrackId \
       ORDER BY steviloProdaj DESC, pesem ASC \
       LIMIT    100",
        (napaka, vrstice) => {
            if (napaka) odgovor.sendStatus(500);
            else {
                // odkomentiraj za reševanje 2. naloge
                for (let i = 0; i < vrstice.length; i++) {
                    vrstice[i].stopnja = davcnaStopnja(
                        vrstice[i].izvajalec,
                        vrstice[i].zanr
                    );
                    vrstice[i].cena = (vrstice[i].cena * (1 + vrstice[i].steviloProdaj / 100)).toFixed(2);
                    
                    // 3. naloga
                    vrstice[i].ozadje= "background-color: linear-gradient(to right, lightgray, " +
                                         "white, whitesmoke, white, " + pridobiBarvoZanra(vrstice[i].zanr) + ");";
                }
                odgovor.render("seznam", {seznamPesmi: vrstice});
            }
        }
    );
});

// Dodajanje oz. brisanje pesmi iz košarice
streznik.get("/kosarica/:idPesmi", (zahteva, odgovor) => {
    let idPesmi = parseInt(zahteva.params.idPesmi);
    if (!zahteva.session.kosarica) zahteva.session.kosarica = [];
    if (zahteva.session.kosarica.indexOf(idPesmi) > -1) {
        // Če je pesem v košarici, jo izbrišemo
        zahteva.session.kosarica.splice(
            zahteva.session.kosarica.indexOf(idPesmi),
            1
        );
    } else {
        // Če pesmi ni v košarici, jo dodamo
        zahteva.session.kosarica.push(idPesmi);
    }
    // V odgovoru vrnemo vsebino celotne košarice
    odgovor.send(zahteva.session.kosarica);
});

// Vrni podrobnosti pesmi v košarici iz podatkovne baze
const pesmiIzKosarice = (zahteva, povratniKlic) => {
    // Če je košarica prazna
    if (!zahteva.session.kosarica || zahteva.session.kosarica.length == 0) {
        povratniKlic([]);
    } else {
        // odkomentiraj za reševanje 1. naloge
        const podatek = "Genre.Name AS zanr,";
        const tabela = ", Genre";
        const pogoj = "Track.GenreId = Genre.GenreId AND ";

        //stevilkaArtikla
        //kolicina
        //zanr
        //cena
        //opisArtikla
        //popust
        //stopnja

        // Sicer dostopaj do podatkovne baze in pridobi podrobnosti
        pb.all(
            "SELECT Track.TrackId AS stevilkaArtikla, \
                    1 AS kolicina, \
                    Track.Name || ' (' || Artist.Name || ')' AS opisArtikla, \
                    ROUND(Track.UnitPrice * " +
            razmerje_USD_EUR +
            ", 2) AS cena, \
                   " + podatek + " \
              0 AS popust \
       FROM   Track, Album, Artist" + tabela + " \
       WHERE  Track.AlbumId = Album.AlbumId AND \
              Artist.ArtistId = Album.ArtistId AND \
              " + pogoj + " \
              Track.TrackId IN (" + zahteva.session.kosarica.join(",") + ")",
            (napaka, vrstice) => {
                if (napaka) povratniKlic(false);
                else {
                    // odkomentiraj za reševanje 2. naloge
                    for (let i = 0; i < vrstice.length; i++) {
                        vrstice[i].stopnja = davcnaStopnja(
                            vrstice[i].opisArtikla.split(" (")[1].split(")")[0],
                            vrstice[i].zanr
                        );
                    }
                    povratniKlic(vrstice);
                }
            }
        );
    }
};

// Podrobnosti košarice
streznik.get("/kosarica", (zahteva, odgovor) => {
    pesmiIzKosarice(zahteva, (pesmi) => {
        if (!pesmi) odgovor.sendStatus(500);
        else odgovor.send(pesmi);
    });
});

// Izpis računa v HTML predstavitvi ali izvorni XML obliki
streznik.get("/izpisiRacun/:oblika", (zahteva, odgovor) => {
    pesmiIzKosarice(zahteva, (pesmi) => {
        if (!pesmi) {
            odgovor.sendStatus(500);
        } else if (pesmi.length == 0) {
            odgovor.send(
                "<p>V košarici nimate nobene pesmi, zato računa ni mogoče pripraviti!</p>"
            );
        } else {
            // odkomentiraj za reševanje 1. naloge
            let zanri = {};
            let steviloPesmi = 0;
            for (let i = 0; i < pesmi.length; i++) {
                // Število pesmi po žanrih
                if (pesmi[i].zanr in zanri) zanri[pesmi[i].zanr]++;
                else zanri[pesmi[i].zanr] = 1;
                // Število pesmi izbranih izvajalcev
                if (
                    pesmi[i].opisArtikla.endsWith("(Iron Maiden)") ||
                    pesmi[i].opisArtikla.endsWith("(Body Count)")
                )
                    steviloPesmi++;
            }
            let skupniPopust =
                (pesmi.length >= 5 ? 20: 0) + // +20 % za več kot 5 pesmi
                (steviloPesmi > 1 ? 5 : 0) + // +5 % za več pesmi izbranih izvajalcev
                (new Date().getMinutes() <= 30 ? 1 : 0); // +1 % za prvo polovico ure
            // Nakup več pesmi istega žanra
            for (let zanr in zanri) {
                if (zanri[zanr] >= 3) {
                    skupniPopust += 10; // +10 % za več kot 3 pesmi žanra
                    break;
                }
            }

            let povzetek = {
                vsotaSPopustiInDavki: 0,
                vsoteZneskovDdv: {0: 0, 9.5: 0, 22: 0, skupaj: 0},
                vsoteOsnovZaDdv: {0: 0, 9.5: 0, 22: 0, skupaj: 0},
                vsotaVrednosti: 0,
                vsotaPopustov: 0,
            };
            let ODGOVOR = 0;

            pesmi.forEach(function (pesem, i) {
                pesem.zapSt = i + 1;
                pesem.vrednost = pesem.kolicina * pesem.cena;
                pesem.davcnaStopnja = 22;
                // odkomentiraj za reševanje 2. naloge
                pesem.davcnaStopnja = pesem.stopnja;
                pesem.popustStopnja = pesem.popust + pesem.stopnja;
                pesem.popust =
                    pesem.kolicina * pesem.cena * (pesem.popustStopnja / 100);
                pesem.osnovaZaDdv = pesem.vrednost - pesem.popust;
                pesem.ddv = pesem.osnovaZaDdv * (pesem.davcnaStopnja / 100);
                pesem.osnovaZaDdvInDdv = pesem.osnovaZaDdv + pesem.ddv;

                povzetek.vsotaSPopustiInDavki += pesem.osnovaZaDdv + pesem.ddv;
                povzetek.vsoteZneskovDdv[pesem.davcnaStopnja] += pesem.ddv;
                povzetek.vsoteZneskovDdv["skupaj"] += pesem.ddv;
                povzetek.vsoteOsnovZaDdv[pesem.davcnaStopnja] += pesem.osnovaZaDdv;
                povzetek.vsoteOsnovZaDdv["skupaj"] += pesem.osnovaZaDdv;
                povzetek.vsotaVrednosti += pesem.vrednost;
                povzetek.vsotaPopustov += pesem.popust;
            });

            odgovor.setHeader("Content-Type", "text/xml");
            odgovor.render("eslog", {
                vizualiziraj: zahteva.params.oblika == "html",
                postavkeRacuna: pesmi,
                povzetekRacuna: povzetek,
            });
        }
    });
});

// Privzeto izpiši račun v HTML obliki
streznik.get("/izpisiRacun", (zahteva, odgovor) => {
    odgovor.redirect("/izpisiRacun/html");
});

streznik.listen(process.env.PORT, () => {
    console.log("Strežnik je pognan!");
});
