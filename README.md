# Notebooks

## CPV Descriptive

Contains stats cpv division, categories, groups and corresponding invitations to tender

## Each Attribute alone

Every attribute tries to predict if an invitation to tender is interesting

# Log of JSON

## dbs/gatt_wto

attributes = ['gatt_wto','lose','teilangebote', 'varianten','sprache']

## dbs/auftragsart_art

attributes = ['auftragsart_art','beschaffungsstelle_plz','gatt_wto','lose','teilangebote', 'varianten','sprache']

## dbs/plz

attributes = ['beschaffungsstelle_plz','gatt_wto','lose','teilangebote', 'varianten','sprache']


# Query used to get most of the needed stuff

```
SELECT *
FROM beruecksichtigteanbieter_zuschlag
JOIN zuschlag ON zuschlag.meldungsnummer = beruecksichtigteanbieter_zuschlag.meldungsnummer
JOIN anbieter ON beruecksichtigteanbieter_zuschlag.anbieter_id = anbieter.anbieter_id
JOIN projekt ON zuschlag.projekt_id = projekt.projekt_id
JOIN auftraggeber ON projekt.auftraggeber_id = auftraggeber.auftraggeber_id
JOIN ausschreibung ON projekt.projekt_id = ausschreibung.projekt_id
JOIN cpv_dokument ON cpv_dokument.meldungsnummer = ausschreibung.meldungsnummer
ORDER BY ausschreibung.meldungsnummer;

```
