const sqlite3 = require("sqlite3").verbose();
const pb = new sqlite3.Database("Chinook.sl3");

// Prikaži vse podatke v tabeli
pb.each("SELECT * FROM Genre", (napaka, vrstica) => {
  if (!napaka) console.log("Žanr : " + vrstica.Name);
  else console.log("Prišlo je do napake!");
});

// Prikaži podatke o izbrani pesmi
pb.each("SELECT Name FROM Track WHERE TrackId=17", (napaka, vrstica) => {
  if (!napaka) console.log("Pesem : " + vrstica.Name);
  else console.log("Prišlo je do napake!");
});

// Prikaži podatke iz več tabel hkrati
pb.each(
  "SELECT  Genre.Name AS zanr, Track.Name AS pesem, \
           Track.Composer AS skladatelj \
  FROM     Track, Genre \
  WHERE    Track.GenreId = Genre.GenreId \
  ORDER BY zanr, pesem \
  LIMIT    20",
  (napaka, vrstica) => {
    if (!napaka)
      console.log(
        "Žanr = " +
          vrstica.zanr +
          ", pesem = " +
          vrstica.pesem +
          (vrstica.skladatelj != null
            ? ", skladatelj = " + vrstica.skladatelj
            : "")
      );
    else console.log("Prišlo je do napake!");
  }
);

pb.close();
