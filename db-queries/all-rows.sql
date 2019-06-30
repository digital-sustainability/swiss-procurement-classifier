SELECT *
FROM beruecksichtigteanbieter_zuschlag
JOIN zuschlag ON zuschlag.meldungsnummer = beruecksichtigteanbieter_zuschlag.meldungsnummer
JOIN anbieter ON beruecksichtigteanbieter_zuschlag.anbieter_id = anbieter.anbieter_id
JOIN projekt ON zuschlag.projekt_id = projekt.projekt_id
JOIN auftraggeber ON projekt.auftraggeber_id = auftraggeber.auftraggeber_id
JOIN ausschreibung ON projekt.projekt_id = ausschreibung.projekt_id
;
