var ime = "Dejan";
var priimek = "Lavbič";

console.log("A ni lep dan, " + ime + " " + priimek);

console.log(13.2 === 13.2);
console.log("10" === 10);
console.log(2 == "2");
console.log(10 < 5);

var ocena = 49.8;
if (ocena >= 49.5) {
  console.log("Izpit ste uspešno opravili!");
} else {
  console.log("Na žalost izpit ni opravljen!");
}

var i = 1;
while (i < 3) {
  console.log("while " + i);
  i = i + 1;
}
for (var i = 1; i < 3; i++) {
  console.log("for " + i);
}

var siOpravilIzpit = function (ocena) {
  return ocena >= 49.5 ? true : false;
};
console.log(siOpravilIzpit(49));
console.log(siOpravilIzpit(50));

var student = {
  ime: "Dejan",
  priimek: "Lavbič",
  izpit: 49,
  predstaviSe: function () {
    console.log(
      "Sem " +
        this.ime +
        " " +
        this.priimek +
        " in sem pri izpitu iz OIS dosegel " +
        this.izpit +
        " točk."
    );
  },
};
student.predstaviSe();

var seznamZelja = ["anakonda", "kobra"];
seznamZelja.push("piton");
console.log(seznamZelja);
console.log(seznamZelja[1]);
seznamZelja.pop();
console.log(seznamZelja);

window.addEventListener("load", function () {
  console.log("Stran se je v celoti naložila.");

  var kontakt = document.getElementById("kontakt");
  console.log(kontakt);

  var telefonskaStevilka = document.querySelector(".telefonska-stevilka");
  console.log(telefonskaStevilka);

  var obdelajKlik = function () {
    console.log("Potrebno je preveriti vnesene podatke!");
  };
  var gumb = document.querySelector("input[type='button']");
  gumb.addEventListener("click", obdelajKlik);

  var req = new XMLHttpRequest();
  req.onload = function () {
    console.log("Podatki oddaljene storitve pridobljeni!");
    console.log(this.responseText);
  };
  req.open(
    "get",
    "https://teaching.lavbic.net/api/kraji/iskanje/postnaStevilka/3000",
    true
  );
  req.send();

  var anakondaNiz = JSON.stringify({
    kaca: "Anakonda",
    habitat: "Južna Amerika",
    dolzina: "5-7 m",
  });
  console.log(anakondaNiz);
  var anakondaObjekt = JSON.parse(anakondaNiz);
  console.log(anakondaObjekt.dolzina);

  var narediNekaj = function () {
    var a = 10;
  };
  narediNekaj();
  //console.log(a); // a ni opredeljen, pride do napake

  var b = 10;
  if (b > 5) {
    var c = 5;
  }
  console.log(b + c);
});
