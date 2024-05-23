SELECT i.id AS 'id', i.name AS 'ime', i.legal_name AS 'pravno ime'
FROM institutions i
INNER JOIN edocument_versions_institutions evi ON i.id = evi.institution_id
WHERE evi.status = 'not_signed'
ORDER BY i.id;

SELECT u.id AS 'id uporabnika', u.email AS 'email', er.role AS 'profil', i.id AS 'id vrtca', i.name AS 'ime vrtca', i.legal_name AS 'pravno ime vrtca'
FROM users u
INNER JOIN employees e ON u.id = e.user_id AND e.deleted_at IS NULL
INNER JOIN employee_roles er ON e.id = er.employee_id AND er.deleted_at IS NULL
INNER JOIN institutions i ON e.institution_id = i.id
INNER JOIN edocument_versions_institutions evi ON i.id = evi.institution_id
WHERE
    evi.status = 'not_signed' AND
    er.role IN ('admin', 'headmaster', 'assistant-headmaster') AND
    u.active = 1 AND
    u.email != '' AND
    u.email IS NOT NULL
ORDER BY i.id;