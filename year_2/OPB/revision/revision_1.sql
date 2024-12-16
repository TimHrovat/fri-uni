-- 1. podatki o zaposlenih, ki imajo drugo najvišjo plačo
SELECT *
FROM ZAPOSLENI
WHERE Placa = (
    SELECT MAX(Placa)
    FROM ZAPOSLENI
    WHERE Placa != (
        SELECT MAX(Placa)
        FROM ZAPOSLENI
    )
);

SELECT *
FROM ZAPOSLENI z
INNER JOIN DELO AS d ON d.ID_delo = z.ID_delo
WHERE d.Funkcija IN ("MANAGER", "CLERK");

SELECT z.Ime, z.Priimek, odd.Ime
FROM ZAPOSLENI z
INNER JOIN ODDELEK AS odd ON odd.ID_oddelek = z.ID_oddelek
INNER JOIN LOKACIJA AS l ON l.ID_lokacija = odd.ID_lokacija
WHERE l.Regija = "DALLAS";

SELECT z1.Ime, z1.Priimek
FROM ZAPOSLENI z1
INNER JOIN ZAPOSLENI AS z2 ON z1.ID_zaposleni = z2.ID_nadrejeni
GROUP BY z1.ID_zaposleni
HAVING count(z2.*) > 0;
