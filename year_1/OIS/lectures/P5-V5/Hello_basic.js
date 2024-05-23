if (!process.env.PORT) process.env.PORT = 8080;

const http = require("http");

const streznik = http.createServer((zahteva, odgovor) => {
  odgovor.writeHead(200, { "Content-Type": "text/plain" });
  odgovor.end("Lepo pozdravljeni ljubitelji predmeta OIS!");
});

streznik.listen(process.env.PORT, () => {
  console.log("Strežnik je pognan!");
});
