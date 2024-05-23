if (!process.env.PORT) process.env.PORT = 3000;

const express = require("express");
const streznik = express();

streznik.use(express.static("public"));

streznik.get("/", (zahteva, odgovor) => {
  odgovor.send("Lepo pozdravljeni ljubitelji predmeta OIS!");
});

streznik.listen(process.env.PORT, () => {
  console.log(`Strežnik je pognan! ${process.env.PORT}`);
});
