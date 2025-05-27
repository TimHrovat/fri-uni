$(document).ready(() => {
    $("select#seznamRacunov").change(function (e) {
        let izbranRacunId = $(this).val();
        $("input#podrobnostiIzleta").attr("racun", izbranRacunId);
        console.log(izbranRacunId);
    });

    $("select#seznamStrank").change(function (e) {
        let izbranaStrankaId = $(this).val();
        $.get("/vec-o-stranki-api/" + izbranaStrankaId, (podatki) => {
            document.getElementById("podrobnostiIzbraneStranke").innerHTML = podatki;
        });
    });
});
