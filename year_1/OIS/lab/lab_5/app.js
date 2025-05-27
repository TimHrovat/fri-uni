if (!process.env.PORT) process.env.PORT = 3000;

/* Initialization of Chinook database */
const sqlite3 = require("sqlite3").verbose();
const db = new sqlite3.Database("Chinook.sl3");

const ITEMS_PER_PAGE = 20;

// Calls callback with specified page's artists and artist's details
const artists = (page, artistSelected, details, callback) => {
    db.all("SELECT Artist.ArtistId, Name, StarsNo " +
        "FROM Artist, Stars WHERE Artist.ArtistId = Stars.ArtistId " +
        "ORDER BY Name LIMIT " + ITEMS_PER_PAGE + " OFFSET  ($page - 1) * " + ITEMS_PER_PAGE,
        {$page: page}, (error, rows) => {
            if (error) {
                console.log(error);
                callback("Something went wrong!");
            } else {
                let artists = [];
                for (let i = 0; i < rows.length; i++) {
                    let artist = {};
                    artist.id = rows[i].ArtistId;
                    artist.number= (page * ITEMS_PER_PAGE + i - (ITEMS_PER_PAGE - 1));
                    artist.page = page;
                    artist.name= rows[i].Name;
                    let isSelected = rows[i].ArtistId == artistSelected;
                    artist.buttonState = isSelected ? "selected" : "";
                    artist.url = (isSelected ? ("/artists/" + artist.page) : ("/artists/" + artist.page + "/details/" + artist.id)) + "#" + artist.id;

                    artist.starsNo = rows[i].StarsNo;
                    artist.stars = [];
                    artist.starsEmpty = [];

                    // when the artist is rated with 1 star
                    if (artist.starsNo == 1) {
                        artist.stars.push({number: 0});
                    }
                    // when the artist is rated with more than 1 star
                    else {
                        for (let j = 0; j < artist.starsNo; j++) {
                            artist.stars.push({number: (j + 1)});
                        }
                    }

                    // add empty stars
                    for (let j = artist.starsNo; j < 3; j++) {
                        artist.starsEmpty.push({number: (j + 1)});
                    }

                    if (rows[i].ArtistId == artistSelected)
                        artist.details = details;

                    artists.push(artist);
                }
                callback(artists);
            }
        }
    );
};


// Calls callback with specified artist's albums
const albums = (artist, callback) => {
    db.all(
        "SELECT * \
        FROM    Album \
        WHERE   ArtistId = $artist \
        ORDER BY Title",
        {$artist: artist},
        function (error, rows) {
            if (error) {
                console.log(error);
                callback("Something went wrong!");
            } else {
                var result = "";
                if (rows.length == 0) {
                    result += "No albums found for this artist";
                } else {
                    rows.forEach(function (row) {
                        result +=
                            "<div album='" + row.AlbumId + "'><em>'" + row.Title +
                            "'</em><span><span class='fas fa-info-circle'></span></span> " +
                            "<a style='color: blue' href='/album-details/" + row.AlbumId +
                            "' target='_blank'><i class='fa-solid fa-bars'></i></a>" +
                            "</div>";
                    });
                }
                callback(result);
            }
        }
    );
};

// Calls callback with specified artist's playlists
const playlists = (artist, callback) => {
    db.all(
        "SELECT DISTINCT Playlist.PlaylistId, Playlist.Name \
        FROM  Playlist, PlaylistTrack, Track, Album \
        WHERE Playlist.PlaylistId = PlaylistTrack.PlaylistId AND \
              PlaylistTrack.TrackId = Track.TrackId AND \
              Track.AlbumId = Album.AlbumId AND \
              ArtistId = $artist \
        ORDER BY Playlist.PlaylistId",
        {$artist: artist},
        function (error, rows) {
            if (error) {
                callback("Something went wrong!");
            } else {
                var result = "";
                if (rows.length == 0) result += "This artist is on no playlists";
                else
                    rows.forEach(function (row) {
                        result +=
                            "<div playlist='" +
                            row.PlaylistId +
                            "'> \
                              <em>" +
                            row.Name +
                            "</em> \
                              <span><span class='fas fa-info-circle'></span></span>\
                            </div>";
                    });
                callback(result);
            }
        }
    );
};

// Calls callback with specified artist's genres
const genres = (artist, callback) => {
    db.all(
        "SELECT   DISTINCT Genre.Name \
         FROM     Genre, Track, Album \
         WHERE    Genre.GenreId = Track.GenreId AND \
                  Track.AlbumId = Album.AlbumId AND \
                  ArtistId = $artist \
         ORDER BY Genre.Name",
        {$artist: artist},
        function (error, rows) {
            if (error) {
                callback("Something went wrong!");
            } else {
                let result = "No genres for this artist";
                // odkomentiraj za reševanje 3. naloge
                if (rows.length == 0) {
                    result += "No genres for this artist";
                } else {
                    result = "";
                    for (let i = 0; i < rows.length; i++) {
                        result += (i > 0 ? " | " : "") + rows[i].Name;
                    }
                }
                callback(result);
            }
        }
    );
};

// Initialization of Express application
const express = require("express");
const app = express();

// set Handlebars view engine
app.set("view engine", "hbs");

// Settings for static application files
app.use(express.static("public"));

// Responds with first page's artists
app.get("/artists", (request, response) => {
    response.redirect("/artists/1");
});

// Responds with specified page's artists
app.get("/artists/:page", (request, response) => {
    artists(request.params.page, -1, "", (result) => {
        response.render("index", {title: "Chinook Artists", artists: result});
    });
});

// Responds with specified page's artists and artist's details
app.get("/artists/:page/details/:artist", (request, response) => {
    albums(request.params.artist, (albums) => {
        playlists(request.params.artist, (playlists) => {
            genres(request.params.artist, (genres) => {
                artists(
                    request.params.page,
                    request.params.artist,
                    {albums: albums, playlists: playlists, genres: genres},
                    (result) => {
                        response.render("index", {title: "Chinook Artists", artists: result});
                    }
                );
            });
        });
    });
});

// Responds with specified artist's albums
app.get("/albums/:artist", (request, response) => {
    albums(request.params.artist, (result) => {
        response.send(result);
    });
});

// Responds with specified album's details
app.get("/album/:album", (request, response) => {
    db.get(
        "SELECT COUNT(*) AS Tracks, \
                SUM(Milliseconds) AS Time, \
                SUM(UnitPrice) AS Price \
         FROM   Track \
         WHERE  AlbumId = $album",
        {$album: request.params.album},
        (error, row) => {
            if (error) {
                response.sendStatus(404);
            } else {
                response.send({
                    tracks: row.Tracks,
                    time: row.Time,
                    price: row.Price,
                });
            }
        }
    );
});

// Responds with specified playlist's details
app.get("/playlist/:playlist", (request, response) => {
    db.get(
        "SELECT COUNT(*) AS Tracks, \
                COUNT(DISTINCT ArtistId) AS Artists, \
                SUM(Milliseconds) AS Time, \
                SUM(UnitPrice) AS Price \
        FROM    Track, PlaylistTrack, Album \
        WHERE   Track.AlbumId = Album.AlbumId AND \
                Track.TrackId = PlaylistTrack.TrackId AND \
                PlaylistId = $playlist",
        {$playlist: request.params.playlist},
        (error, row) => {
            if (error) {
                response.sendStatus(404);
            } else {
                response.send({
                    tracks: row.Tracks,
                    artists: row.Artists,
                    time: row.Time,
                    price: row.Price,
                });
            }
        }
    );
});

// Updates and responds with specified artist's stars
app.get("/stars/:artist/:stars", (request, response) => {
    db.get(
        "UPDATE  Stars \
         SET     StarsNo = $stars \
         WHERE   ArtistId = $artist",
        {
            $artist: request.params.artist,
            $stars: request.params.stars,
        },
        (error, row) => {
            if (error) {
                response.sendStatus(404);
            } else {
                response.send({
                    artist: request.params.artist,
                    stars: request.params.stars,
                });
            }
        }
    );
});

// Responds with number of artist pages
app.get("/pages", (request, response) => {
    db.get("SELECT COUNT(*) AS Artists FROM Artist", (error, row) => {
        if (error) {
            response.sendStatus(500);
        } else {
            response.send({
                pages: Math.ceil(row.Artists / ITEMS_PER_PAGE),
            });
        }
    });
});

// Respond with album details with a Handlebars view
app.get("/album-details/:album", (request, response) => {
    db.all("SELECT *, Artist.Name AS ArtistName, Track.Name AS TrackName, Track.UnitPrice AS PriceUSD, Album.Title AS AlbumTitle,\
           MediaType.Name AS MediaTypeName \
         FROM   Track, Album, Artist, MediaType \
         WHERE  Album.AlbumId = Track.AlbumId AND \
                Album.ArtistId = Artist.ArtistId AND \
                Track.MediaTypeId = MediaType.MediaTypeId \
                AND Track.AlbumId = $album",
        {$album: request.params.album},
        (error, row) => {
            if (error) {
                response.sendStatus(500);
            } else {
                for (var i = 0; i < row.length; i++) {
                    let seconds = row[i].Milliseconds / 1000;
                    row[i].Minutes = seconds / 60;
                    row[i].Seconds = (seconds - (parseInt(row[i].Minutes) * 60)).toFixed(0);
                    row[i].Minutes = row[i].Minutes.toFixed(0);

                    row[i].MB = ((row[i].Bytes / 1024) / 1024).toFixed(0);
                }

                response.render("album-details", {
                    title: "Album details",
                    artist: row.length > 0 ? row[0].ArtistName : "",
                    albumTitle: row.length > 0 ? row[0].AlbumTitle : "",
                    tracks: row
                });
            }
        });
});

// Responds with first page's artists
// odkomentiraj za reševanje 1. naloge
app.get("/", (request, response) => {
    response.redirect("/artists/1");
});

// Starts application on set port
app.listen(process.env.PORT, () => {
    console.log("App listening on port " + process.env.PORT + ".");
});
