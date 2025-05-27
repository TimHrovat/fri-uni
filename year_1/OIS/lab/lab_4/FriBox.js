if (!process.env.PORT) {
  process.env.PORT = 8080;
}

var mime = require("mime-types");
var formidable = require("formidable");
var http = require("http");
var fs = require("fs-extra");
var util = require("util");
var path = require("path");

var dataDir = "./data/";

var streznik = http.createServer(function (zahteva, odgovor) {
  if (zahteva.url == "/") {
    posredujOsnovnoStran(odgovor);
  } else if (zahteva.url == "/datoteke") {
    posredujSeznamDatotek(odgovor);
  } else if (zahteva.url.startsWith("/brisi")) {
    izbrisiDatoteko(odgovor, dataDir + zahteva.url.replace("/brisi", ""));
  } else if (zahteva.url.startsWith("/prenesi")) {
    posredujStaticnoVsebino(
      odgovor,
      dataDir + zahteva.url.replace("/prenesi", ""),
      "application/octet-stream"
    );
  } else if (zahteva.url == "/nalozi") {
    naloziDatoteko(zahteva, odgovor);
  } else if (zahteva.url.startsWith('/poglej')) {
    // odkomentiraj za reševanje 5. naloge
    posredujStaticnoVsebino(odgovor, dataDir + zahteva.url.replace("/poglej", ""), "");
  } else {
    posredujStaticnoVsebino(odgovor, "./public" + zahteva.url, "");
  }
});

function posredujOsnovnoStran(odgovor) {
  posredujStaticnoVsebino(odgovor, "./public/fribox.html", "");
}

function posredujStaticnoVsebino(odgovor, absolutnaPotDoDatoteke, mimeType) {
  fs.exists(absolutnaPotDoDatoteke, function (datotekaObstaja) {
    if (datotekaObstaja) {
      fs.readFile(absolutnaPotDoDatoteke, function (napaka, datotekaVsebina) {
        if (napaka) {
          posredujNapako500(odgovor);
        } else {
          posredujDatoteko(
            odgovor,
            absolutnaPotDoDatoteke,
            datotekaVsebina,
            mimeType
          );
        }
      });
    } else {
      posredujNapako404(odgovor);
    }
  });
}

function posredujDatoteko(odgovor, datotekaPot, datotekaVsebina, mimeType) {
  if (mimeType == "") {
    odgovor.writeHead(200, {
      "Content-Type": mime.lookup(path.basename(datotekaPot)),
    });
  } else {
    odgovor.writeHead(200, { "Content-Type": mimeType });
  }

  odgovor.end(datotekaVsebina);
}

function posredujSeznamDatotek(odgovor) {
  odgovor.writeHead(200, { "Content-Type": "application/json" });
  fs.readdir(dataDir, function (napaka, datoteke) {
    if (napaka) {
      posredujNapakoODGOVOR(odgovor);
    } else {
      var rezultat = [];
      for (var i = 0; i < datoteke.length; i++) {
        var datoteka = datoteke[i];
        var velikost = fs.statSync(dataDir + datoteka).size;
        rezultat.push({ datoteka: datoteka, velikost: velikost });
      }

      odgovor.write(JSON.stringify(rezultat));
      odgovor.end();
    }
  });
}

function naloziDatoteko(zahteva, odgovor) {
    var form = new formidable.IncomingForm();

    form.parse(zahteva, function (napaka, polja, datoteke) {
        util.inspect({fields: polja, files: datoteke});
    });

    form.on("end", function (fields, files) {
        var zacasnaPot = this.openedFiles[0].filepath;
        var datoteka = this.openedFiles[0].originalFilename;
        // odkomentiraj za reševanje 8. naloge
        fs.readdir(dataDir, function (napakaBranja, datoteke) {
            if (napakaBranja) {
                posredujNapako500(odgovor);
            } else {
                var enakoIme = false;
                for (var i = 0; i < datoteke.length; i++) {
                    var imeDatoteke = datoteke[i];
                    if (datoteka == imeDatoteke) {
                        enakoIme = true;
                        break;
                    }
                }
                if (enakoIme) {
                    posredujNapako409(odgovor);
                } else {
                    fs.copy(zacasnaPot, dataDir + datoteka, function (napaka) {
                        if (napaka) {
                            posredujNapako500(odgovor);
                        } else {
                            posredujOsnovnoStran(odgovor);
                        }
                    });
                    // ne pozabite izbrisati podvojenega posredovanja odgovora
                }
            }
        });
        fs.copy(zacasnaPot, dataDir + datoteka, function (napaka) {
            if (napaka) {
                posredujNapakoODGOVOR(odgovor);
            } else {
                posredujOsnovnoStran(odgovor);
            }
        });
    });
}

function izbrisiDatoteko(odgovor, datoteka) {
  odgovor.writeHead(200, {'Content-Type': 'text/plain'});
  fs.unlink(datoteka, function(napaka) {
    if (napaka) {
      posredujNapakoODGOVOR(odgovor);
    } else {
      odgovor.write("Datoteka izbrisana");
      odgovor.end();
    }
  });
}

//  Napake
function posredujNapako404(odgovor) {
  odgovor.writeHead(404, {'Content-Type': 'text/plain'});
  odgovor.write('Napaka 404: Vira ni mogoče najti!');
  odgovor.end();
}

function posredujNapako500(odgovor) {
  odgovor.writeHead(500, {'Content-Type': 'text/plain'});
  odgovor.write('Napaka 500: Prišlo je do napake strežnika.');
  odgovor.end();
}

function posredujNapako409(odgovor) {
  odgovor.writeHead(409, {'Content-Type': 'text/plain'});
  odgovor.write('Napaka 409: Datoteka s podanim imenom ODGOVOR obstaja.');
  odgovor.end();
}

streznik.listen(process.env.PORT, function() {
  console.log("Strežnik je pognan.");
});
