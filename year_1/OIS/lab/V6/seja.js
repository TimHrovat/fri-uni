if (!process.env.PORT) process.env.PORT = 3000;

const express = require("express");
const streznik = express();

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
      // Seja poteče po 2 s neaktivnost
      maxAge: 2000,
    },
  })
);

let stDostopov = 0;

streznik.get("/", (zahteva, odgovor) => {
  stDostopov++;
  if (!zahteva.session.stDostopov) zahteva.session.stDostopov = 1;
  else zahteva.session.stDostopov++;
  odgovor.send(
    "<h1>Globalne spremenljivke vs. seja</h1>" +
      "<p style='font-size:150%'>" +
      "To je že <strong>" +
      stDostopov +
      "x</strong> dostop do strežnika, " +
      "medtem ko ste v tej seji dostopali <strong>" +
      zahteva.session.stDostopov +
      "x</strong>." +
      "</p>"
  );
});

streznik.listen(process.env.PORT, () => {
  console.log("Strežnik je pognan!");
});
