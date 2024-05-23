if (!process.env.PORT) process.env.PORT = 3000;

const express = require("express");
const streznik = express();

const sqlite3 = require("sqlite3").verbose();
const pb = new sqlite3.Database("Chinook.sl3");

/**
 * Vrni seznam žanrov v povratnem klicu
 *
 * @param povratniKlic rezultat klica funkcije, ki predstavlja tabelo vrstic
 * in v vsaki vrstici je podatek o nazivu žanra
 */
const vrniVseZanre = (povratniKlic) => {
  pb.all("SELECT * FROM Genre", (napaka, vrstice) => {
    let rezultat = "Prišlo je do napake!";
    if (!napaka) {
      rezultat = "<h1>" + "Žanri" + "</h1>" + "<ul>";
      vrstice.forEach((vrstica) => {
        rezultat +=
          "<li>" +
          "<a href='/kategorija/" +
          vrstica.GenreId +
          "'>" +
          vrstica.Name +
          "</a>" +
          "</li>";
      });
      rezultat += "</ul>";
    }
    povratniKlic(rezultat);
  });
};

/**
 * Vrni vse pesmi podanega žanra v povratnem klicu
 *
 * @param idZanra ID žanra
 * @param povratniKlic rezultat klica funkcije, ki predstavlja HTML kodo
 * v obliki alinej, kjer je v vsaki podatek o nazivu in ceni pesmi
 */
const vrniPesmiZanra = (idZanra, povratniKlic) => {
  pb.all(
    "SELECT * FROM Track WHERE GenreId = $id",
    { $id: idZanra },
    (napaka, vrstice) => {
      let rezultat = "Prišlo je do napake!";
      if (!napaka) {
        rezultat = "<h2>" + "Pesmi" + "</h2>" + "<ul>";
        vrstice.forEach((vrstica) => {
          rezultat +=
            "<li>" +
            "<b>" +
            vrstica.Name +
            "</b>" +
            " @ " +
            "$" +
            vrstica.UnitPrice +
            "</li>";
        });
        rezultat += "</ul>";
      }
      povratniKlic(rezultat);
    }
  );
};

/**
 * Vrni vse pesmi, skupaj z avtorji, podanega žanra v povratnem klicu
 *
 * @param idZanra ID žanra
 * @param povratniKlic rezultat klica funkcije, ki predstavlja HTML kodo
 * v obliki alinej, kjer je v vsaki podatek o nazivu, izvajalcu in ceni
 */
const vrniPesmiInAvtorjeZanra = (idZanra, povratniKlic) => {
  pb.all(
    "SELECT Track.Name AS pesem, Artist.Name AS izvajalec, \
            Track.UnitPrice AS cena \
    FROM    Track, Album, Artist \
    WHERE   Track.AlbumId = Album.AlbumId AND \
            Album.ArtistId = Artist.ArtistId AND \
            GenreId = $id",
    { $id: idZanra },
    (napaka, vrstice) => {
      let rezultat = "Prišlo je do napake!";
      if (!napaka) {
        rezultat = "<h2>" + "Pesmi" + "</h2>" + "<ul>";
        vrstice.forEach((vrstica) => {
          rezultat +=
            "<li>" +
            "<b>" +
            vrstica.pesem +
            "</b>" +
            " (" +
            vrstica.izvajalec +
            ")" +
            " @ " +
            "$" +
            vrstica.cena +
            "</li>";
        });
        rezultat += "</ul>";
      }
      povratniKlic(rezultat);
    }
  );
};

streznik.get("/", (zahteva, odgovor) => {
  vrniVseZanre((rezultat) => {
    odgovor.send(rezultat);
  });
});

streznik.get("/kategorija/:idKategorije", (zahteva, odgovor) => {
  vrniVseZanre((rezultatMaster) => {
    vrniPesmiInAvtorjeZanra(zahteva.params.idKategorije, (rezultatDetail) => {
      odgovor.send(rezultatMaster + rezultatDetail);
    });
  });
});

streznik.listen(process.env.PORT, () => {
  console.log("Strežnik je pognan!");
});
