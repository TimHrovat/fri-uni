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
      // Seja poteče po 60 s (1 min) neaktivnosti
      maxAge: 60000,
    },
  })
);

const razmerje_USD_EUR = 0.84;

// Prikaz seznama pesmi na strani
streznik.get("/", (zahteva, odgovor) => {
  pb.all(
    "SELECT   Track.TrackId AS id, \
              TRACK.Name AS pesem, \
              Artist.Name AS izvajalec, \
              ROUND(Track.UnitPrice * " +
      razmerje_USD_EUR +
      ", 2) AS cena, \
              COUNT(InvoiceLine.InvoiceId) AS steviloProdaj \
     FROM     Track, Album, Artist, InvoiceLine \
     WHERE    Track.AlbumId = Album.AlbumId AND \
              Artist.ArtistId = Album.ArtistId AND \
              InvoiceLine.TrackId = Track.TrackId \
     GROUP BY Track.TrackId \
     ORDER BY steviloProdaj DESC, pesem ASC \
     LIMIT    100",
    (napaka, vrstice) => {
      if (napaka) odgovor.sendStatus(500);
      else odgovor.render("seznam", { seznamPesmi: vrstice });
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
    // Sicer dostopaj do podatkovne baze in pridobi podrobnosti
    pb.all(
      "SELECT Track.TrackId AS stevilkaArtikla, \
              1 AS kolicina, \
              Track.Name || ' (' || Artist.Name || ')' AS opisArtikla, \
              ROUND(Track.UnitPrice * " +
        razmerje_USD_EUR +
        ", 2) AS cena, \
              0 AS popust \
       FROM   Track, Album, Artist \
       WHERE  Track.AlbumId = Album.AlbumId AND \
              Artist.ArtistId = Album.ArtistId AND \
              Track.TrackId IN (" +
        zahteva.session.kosarica.join(",") +
        ")",
      (napaka, vrstice) => {
        if (napaka) povratniKlic(false);
        else povratniKlic(vrstice);
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

streznik.listen(process.env.PORT, () => {
  console.log("Strežnik je pognan!");
});
