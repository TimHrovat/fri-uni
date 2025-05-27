let web3;

/**
 * Funkcija za donacijo Ethereum kriptovalute
 */
const donirajEthereum = async () => {
  try {
    var posiljateljDenarnica = $("#eth-racun").attr("title");
    var prejemnikDenarnica = $("#izbrana-denarnica").val();

    let rezultat = await web3.eth.sendTransaction({
      from: posiljateljDenarnica,
      to: prejemnikDenarnica,
      value: $("#visina-donacije").val() * Math.pow(10, 18),
    });

    // ob uspešni transakciji
    if (rezultat) {
      $("#donacija-odgovor").html(
        "Donacija " + $("#visina-donacije").val() + " ETH je bila uspešna!",
      );
      dopolniTabeloDonacij();
    } else {
      // neuspešna transakcija
      $("#napakaDonacija").html(
        "<div class='alert alert-danger' role='alert'>" +
          "<i class='fas fa-exclamation-triangle me-2'></i>" +
          "Prišlo je do napake pri transakciji!" +
          "</div>",
      );
    }
  } catch (e) {
    // napaka pri transakciji
    $("#napakaDonacija").html(
      "<div class='alert alert-danger' role='alert'>" +
        "<i class='fas fa-exclamation-triangle me-2'></i>" +
        "Prišlo je do napake pri transakciji: " +
        e +
        "</div>",
    );
  }
};

/**
 * Funkcija za prikaz donacij v tabeli
 */
const dopolniTabeloDonacij = async () => {
  let prikaziDonacije = $("#izbrana-denarnica").val().length > 0;
  if (prikaziDonacije) {
    $("#tabela-donacij").show();
    $("#donacije-sporocilo").hide();
  } else {
    $("#tabela-donacij").hide();
    $("#donacije-sporocilo").show();
  }

  try {
    $("#seznam-donacij").html("");
    let steviloBlokov = (await web3.eth.getBlock("latest")).number;
    let st = 1;
    let denarnicaPrejemnika = $("#izbrana-denarnica").val();
    for (let i = 0; i <= steviloBlokov; i++) {
      let blok = await web3.eth.getBlock(i);

      for (let txHash of blok.transactions) {
        let tx = await web3.eth.getTransaction(txHash);
        if (denarnicaPrejemnika && denarnicaPrejemnika == tx.to) {
          $("#seznam-donacij").append(
            "\
                    <tr>\
                        <th scope='row'>" +
              st++ +
              "</th>\
                        <td>" +
              okrajsajNaslov(tx.hash) +
              "</td>\
                        <td>" +
              okrajsajNaslov(tx.from) +
              "</td>\
                        <td>" +
              parseFloat(web3.utils.fromWei(tx.value)) +
              " <i class='fa-brands fa-ethereum'></i></td>\
                    </tr>",
          );
        }
      }
    }
  } catch (e) {
    alert(e);
  }
};

function okrajsajNaslov(vrednost) {
  return (
    vrednost.substring(0, 5) +
    "..." +
    vrednost.substring(vrednost.length - 3, vrednost.length)
  );
}

/**
 * Funkcija za generiranje nove Ethereum denarnice
 */
const ustvariEthereumDenarnico = async () => {
  try {
    let novoGeslo = $("#geslo-ustvari").val();
    let denarnicaUstvarjenegaRacuna =
      await web3.eth.personal.newAccount(novoGeslo);

    if (denarnicaUstvarjenegaRacuna) {
      prijavaEthereumDenarnice(denarnicaUstvarjenegaRacuna, novoGeslo);
    } else {
      $("#napakaPrijava").html(
        "<div class='alert alert-danger' role='alert'>" +
          "<i class='fas fa-exclamation-triangle me-2'></i>" +
          "Prišlo je do napake pri generiranju denarnice!" +
          "</div>",
      );
    }
  } catch (napaka) {
    // napaka pri generiranju denarnice
    $("#napakaPrijava").html(
      "<div class='alert alert-danger' role='alert'>" +
        "<i class='fas fa-exclamation-triangle me-2'></i>" +
        "Prišlo je do napake pri generiranju denarnice: " +
        napaka +
        "</div>",
    );
  }
};

/**
 * Funkcija za prijavo z uporabo prijavne forme
 */
const prijavaPrijavnoOkno = () => {
  prijavaEthereumDenarnice(null, null);
};

/**
 * Funkcija za prijavo Ethereum denarnice v testno omrežje
 */
const prijavaEthereumDenarnice = async (denarnica, geslo) => {
  try {
    let denarnicaPrijava = denarnica
      ? denarnica
      : $("#denarnica-prijava").val();
    let gesloPrijava = geslo ? geslo : $("#geslo-prijava").val();

    let rezultat = await web3.eth.personal.unlockAccount(
      denarnicaPrijava,
      gesloPrijava,
      300,
    );

    // ob uspešni prijavi računa
    if (rezultat) {
      prijavljenRacun = denarnicaPrijava;
      $("#eth-racun").html(
        okrajsajNaslov(denarnicaPrijava) + "<br>(denarnica odklenjena 5 min)",
      );
      // prikažemo celotni naslov ob premiku z miško na HTML element z enoličnim identifikatorjem 'eth-racun'
      $("#eth-racun").attr("title", denarnicaPrijava);
      prikaziKandidateZaDonacije();
      omogociAliOnemogociGumbDoniraj();
      $("#napakaPrijava").html("");
    } else {
      // neuspešna prijava računa
      $("#napakaPrijava").html(
        "<div class='alert alert-danger' role='alert'>" +
          "<i class='fas fa-exclamation-triangle me-2'></i>" +
          "Prišlo je do napake pri odklepanju!" +
          "</div>",
      );
    }
  } catch (napaka) {
    // napaka pri prijavi računa
    $("#napakaPrijava").html(
      "<div class='alert alert-danger' role='alert'>" +
        "<i class='fas fa-exclamation-triangle me-2'></i>" +
        "Prišlo je do napake pri odklepanju: " +
        napaka +
        "</div>",
    );
  }
};

const prikaziKandidateZaDonacije = async () => {
  $("#kandidati").html("");
  let racuni = await web3.eth.personal.getAccounts();
  for (let i in racuni) {
    let racun = racuni[i];
    let onemogoci = "";

    // v denarnico prijavljenega uporabnika nima smisla izvajati donacije
    if (
      $("#eth-racun").attr("title") &&
      racun.toLowerCase() == $("#eth-racun").attr("title").toLowerCase()
    )
      onemogoci = "disabled";

    let stanje = await web3.eth.getBalance(racun);
    $("#kandidati").append(
      parseInt(i) +
        1 +
        ". <input type='radio' name='naslov' value='" +
        racun +
        "' " +
        onemogoci +
        "> \
            <span class='text-muted'>Naslov: </span> <span title='" +
        racun +
        "' naslov='" +
        racun +
        "'>\
            " +
        okrajsajNaslov(racun) +
        " <span class='text-muted'>Stanje: </span> \
            " +
        parseFloat(web3.utils.fromWei(stanje)).toFixed(2) +
        "ETH</span></br>",
    );
  }
};

function omogociAliOnemogociGumbDoniraj() {
  if (
    $("#izbrana-denarnica").val().length > 0 &&
    $("#eth-racun").attr("title") &&
    $("#eth-racun").attr("title").length > 0 &&
    $("#visina-donacije").val().length > 0
  )
    $("#gumb-doniraj-start").removeAttr("disabled");
  else $("#gumb-doniraj-start").attr("disabled", "disabled");
}

function prikaziUstvariPrijavaFormo(formaUstvari) {
  $(formaUstvari ? "#prijava-forma" : "#ustvari-forma").hide();
  $(formaUstvari ? "#ustvari-forma" : "#prijava-forma").show();
}

$(document).ready(function () {
  /* Povežemo se na testno Ethereum verigo blokov */
  web3 = new Web3("http://127.0.0.1:9545");

  /* Dodamo poslušalca na izbirne gumbe (angl. radio buttons)
       kandidatov za donacije */
  $("#kandidati").change(function () {
    let izbranKandidat = $("input[name='naslov']:checked").val();
    $("#izbrana-denarnica").val(izbranKandidat);
    omogociAliOnemogociGumbDoniraj();
    dopolniTabeloDonacij();
  });

  /* Dodamo poslušalca na vnosno polje v 3. koraku */
  $("#visina-donacije").change(function () {
    omogociAliOnemogociGumbDoniraj();
  });

  /* Dodamo poslušalce na vse gumbe v aplikaciji */
  $("#ustvari-racun").click(ustvariEthereumDenarnico);
  $("#prijava-racuna").click(prijavaPrijavnoOkno);
  $("#gumb-doniraj-start").click(donirajEthereum);

  prikaziKandidateZaDonacije();
});
