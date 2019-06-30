SELECT alles.institution, count(*) as count FROM (
    SELECT anbieter.institution, ausschreibung.meldungsnummer
    FROM beruecksichtigteanbieter_zuschlag
    JOIN zuschlag ON zuschlag.meldungsnummer = beruecksichtigteanbieter_zuschlag.meldungsnummer
    JOIN anbieter ON beruecksichtigteanbieter_zuschlag.anbieter_id = anbieter.anbieter_id
    JOIN projekt ON zuschlag.projekt_id = projekt.projekt_id
    JOIN auftraggeber ON projekt.auftraggeber_id = auftraggeber.auftraggeber_id
    JOIN ausschreibung ON projekt.projekt_id = ausschreibung.projekt_id
    WHERE anbieter.institution IS NOT NULL
    GROUP BY ausschreibung.meldungsnummer, anbieter.institution
) alles
GROUP BY alles.institution
ORDER BY COUNT(*) DESC;
