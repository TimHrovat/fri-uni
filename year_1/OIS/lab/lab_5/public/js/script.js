$(document).ready(() => {
    // On page load execute the following code

    // Find selected artist page
    let page = 1;
    let url = $(location).attr("href").split("/artists")[1];
    if (url.length > 0) {
        url = url.split("/")[1];
        if (url.length > 0) page = parseInt(url, 10);
    }

    // Requests and sets artist pages
    $.get("/pages", function (data) {
        for (let i = 0; i < data.pages; i++) {
            $("#pages").append(
                "<a href='/artists/" + (i + 1) + "'>\
                <button type='button' class='btn btn-light btn-sm" +
                (i + 1 == page ? " selected" : "") + "'>" +
                (i + 1) + "</button>\
              </a>"
            );
        }
    });

    // Requests update of artist's stars on click
    $("i[stars]").click(function () {
        $.get(
            "/stars/" +
            $(this).parent().parent().attr("id") +
            "/" +
            $(this).attr("stars"),
            (data) => location.reload()
        );
    });

    // Requests and sets album's details on click
    $("#albums div span").click(function () {
        var details = $(this);
        $.get("/album/" + details.parent().attr("album"), (data) => {
            details.html(
                " (" + data.tracks + " track" + (data.tracks != 1 ? "s" : "") +
                " | " + Math.round(data.time / 60000) + " min | <strong>$" +
                Math.round(data.price) + "</strong> total)"
            );
        });
    });

    // Requests and sets playlist's details on click
    $('#playlists div span').click(function () {
        let details = $(this);
        $.get('/playlist/' + details.parent().attr('playlist'), (data) => {
            details.html(' (' + data.artists + ' artist' + (data.artists != 1 ? 's' : '') + ' | ' +
                data.tracks + ' track' + (data.tracks != 1 ? 's' : '') + ' | ' + '<strong>$' +
                Math.round(data.price) + '</strong> total)');
        });
    });
});
