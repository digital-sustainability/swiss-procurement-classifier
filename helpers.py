from db import connection, engine
import math
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc

# =====================
# SQL SELECT STATEMENTS
# =====================


# @param select: SELECT argument formatted as string
# @return a Pandas dataframe from the full Simap datanbase depending on the SQL SELECT Query
def getFromSimap(select):
    query = """SELECT {} from (((((beruecksichtigteanbieter_zuschlag
        INNER JOIN zuschlag ON zuschlag.meldungsnummer = beruecksichtigteanbieter_zuschlag.meldungsnummer)
        INNER JOIN anbieter ON beruecksichtigteanbieter_zuschlag.anbieter_id = anbieter.anbieter_id)
        INNER JOIN projekt ON zuschlag.projekt_id = projekt.projekt_id)
        INNER JOIN auftraggeber ON projekt.auftraggeber_id = auftraggeber.auftraggeber_id)
        INNER JOIN ausschreibung ON projekt.projekt_id = ausschreibung.projekt_id
        INNER JOIN cpv_dokument ON cpv_dokument.meldungsnummer = zuschlag.meldungsnummer)
        INNER JOIN cpv ON cpv_dokument.cpv_nummer = cpv.cpv_nummer;
    """.format(select)
    return pd.read_sql(query, connection);

# @param bidder: anbieter.institution name formatted as string
# @return a Pandas dataframe showing the most important CPV codes per bidder. (Zuschläge pro CPV Code)
def getCpvCount(bidder):
    query = """SELECT cpv.cpv_nummer, cpv.cpv_deutsch, COUNT(cpv_dokument.cpv_nummer)
        FROM cpv, cpv_dokument, zuschlag, beruecksichtigteanbieter_zuschlag, anbieter WHERE
        cpv.cpv_nummer = cpv_dokument.cpv_nummer AND
        cpv_dokument.meldungsnummer = zuschlag.meldungsnummer AND
        zuschlag.meldungsnummer = beruecksichtigteanbieter_zuschlag.meldungsnummer AND
        beruecksichtigteanbieter_zuschlag.anbieter_id = anbieter.anbieter_id AND
        anbieter.institution = "{}"
        GROUP BY cpv_nummer
        ORDER BY COUNT(cpv_dokument.cpv_nummer) DESC;
    """.format(bidder)
    return pd.read_sql(query, connection);

# @param bidder: anbieter.institution formatted as string of which you want to see the CPV code diversity
# @return a Pandas Dataframe that contains a the diversity of CPV codes per bidder
def getCpvDiversity(bidder):
    query = """SELECT anbieter.institution, COUNT(beruecksichtigteanbieter_zuschlag.anbieter_id)
        AS "Anzahl Zuschläge", COUNT(DISTINCT cpv_dokument.cpv_nummer) AS "Anzahl einzigartige CPV-Codes", 
        SUM(IF(beruecksichtigteanbieter_zuschlag.preis_summieren = 1,beruecksichtigteanbieter_zuschlag.preis,0))
        AS "Ungefähres Zuschlagsvolumen", MIN(zuschlag.datum_publikation) AS "Von", MAX(zuschlag.datum_publikation) AS "Bis"
        FROM cpv, cpv_dokument, zuschlag, beruecksichtigteanbieter_zuschlag, anbieter
        WHERE cpv.cpv_nummer = cpv_dokument.cpv_nummer AND
        cpv_dokument.meldungsnummer = zuschlag.meldungsnummer AND
        zuschlag.meldungsnummer = beruecksichtigteanbieter_zuschlag.meldungsnummer AND
        beruecksichtigteanbieter_zuschlag.anbieter_id = anbieter.anbieter_id
        AND anbieter.institution="{}"
        GROUP BY anbieter.institution
        ORDER BY `Anzahl einzigartige CPV-Codes` DESC
    """.format(bidder)
    return pd.read_sql(query, connection);


# @param select_anbieter: SQL SELECT for the bidder side. Backup:
'''
select_an = (
    "anbieter.anbieter_id, "
    "anbieter.anbieter_plz, "
    "anbieter.institution as anbieter_insitution, "
    "cpv_dokument.cpv_nummer as anbieter_cpv, "
    "ausschreibung.meldungsnummer" )
'''
# @param select_aus: SQL SELECT for the open tenders. Backup:
'''
select_aus = (
    "anbieter.anbieter_id, "
    "auftraggeber.institution as beschaffungsstelle_institution, "
    "auftraggeber.beschaffungsstelle_plz, "
    "ausschreibung.gatt_wto, "
    "cpv_dokument.cpv_nummer as ausschreibung_cpv, "
    "ausschreibung.meldungsnummer" )
'''
# @param bidder: the bidder formatted as string you or do not want the corresponding responses from
# @param response: True if you want all the tenders of the bidder or False if you do not want any (the negative response)
# @return a dataframe containing negative or positive bidding cases of a chosen bidder
def getResponses(select_anbieter, select_ausschreibung, bidder, response):
    resp = '=';
    if (not response):
        resp = '!='
    query = """SELECT * FROM (SELECT {} from ((((((beruecksichtigteanbieter_zuschlag
            INNER JOIN zuschlag ON zuschlag.meldungsnummer = beruecksichtigteanbieter_zuschlag.meldungsnummer)
            INNER JOIN anbieter ON beruecksichtigteanbieter_zuschlag.anbieter_id = anbieter.anbieter_id)
            INNER JOIN projekt ON zuschlag.projekt_id = projekt.projekt_id)
            INNER JOIN auftraggeber ON projekt.auftraggeber_id = auftraggeber.auftraggeber_id)
            INNER JOIN ausschreibung ON projekt.projekt_id = ausschreibung.projekt_id)
            INNER JOIN cpv_dokument ON cpv_dokument.meldungsnummer = zuschlag.meldungsnummer)
            WHERE anbieter.institution {} "{}" ) anbieter
        JOIN (SELECT {} from ((((((beruecksichtigteanbieter_zuschlag
            INNER JOIN zuschlag ON zuschlag.meldungsnummer = beruecksichtigteanbieter_zuschlag.meldungsnummer)
            INNER JOIN anbieter ON beruecksichtigteanbieter_zuschlag.anbieter_id = anbieter.anbieter_id)
            INNER JOIN projekt ON zuschlag.projekt_id = projekt.projekt_id)
            INNER JOIN auftraggeber ON projekt.auftraggeber_id = auftraggeber.auftraggeber_id)
            INNER JOIN ausschreibung ON projekt.projekt_id = ausschreibung.projekt_id)
            INNER JOIN cpv_dokument ON cpv_dokument.meldungsnummer = ausschreibung.meldungsnummer)
            WHERE anbieter.institution {} "{}"
            ) ausschreibung ON ausschreibung.meldungsnummer2 = anbieter.meldungsnummer
        ORDER BY ausschreibung.meldungsnummer2;
    """.format(select_anbieter, resp, bidder, select_ausschreibung, resp, bidder)
    return pd.read_sql(query, connection);


# @return
def getCpvRegister():
    return pd.read_sql("SELECT * FROM cpv", connection);

# @param select_an
# @param select_aus
# @param anbieter
# @return
def createAnbieterDf(select_an, select_aus, anbieter):
    # Create a new DFs one containing all positiv, one all the negative responses
    data_pos = getResponses(select_an, select_aus, anbieter, True)
    data_neg = getResponses(select_an, select_aus, anbieter, False)
    return data_pos.copy(), data_neg.copy()


# ========================
# MODEL CREATION FUNCTIONS
# ========================


# @param df_pos_full
# @param df_neg_full
# @param negSampleSize
# @return
def decisionTreeRun(df_pos_full, df_neg_full , neg_sample_size):
    df_pos = df_pos_full
    # Create a random DF subset ussed to train the model on
    df_neg = df_neg_full.sample(neg_sample_size)
    # Assign pos/neg lables to both DFs
    df_pos['Y']=1
    df_neg['Y']=0
    # Merge the DFs into one
    df_appended = df_pos.append(df_neg, ignore_index=True)
    # Clean PLZ property
    df_appended[['anbieter_plz']] = df_appended[['anbieter_plz']].applymap(tonumeric)
    df_appended[['beschaffungsstelle_plz']] = df_appended[['beschaffungsstelle_plz']].applymap(tonumeric)
    # Shuffle the df
    df_tree = df_appended.sample(frac=1)
    # Put responses in one arry and all diesired properties in another
    y = df_tree.iloc[:,[11]]
    x = df_tree.iloc[:,[1,3,7,9]]
    # create sets
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25)
    # train the model on training sets
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(xtrain, ytrain)
    # predict on the test sets
    res = clf.predict(xtest)
    ytest["res"]= res
    ytest['richtig'] = ytest['res']==ytest['Y']
    tp = ytest[(ytest['Y']==1) & (ytest['res']==1)]
    tn = ytest[(ytest['Y']==0) & (ytest['res']==0)]
    fp = ytest[(ytest['Y']==0) & (ytest['res']==1)]
    fn = ytest[(ytest['Y']==1) & (ytest['res']==0)]
    return len(df_pos.index) / neg_sample_size, accuracy_score(ytest.Y, res), confusion_matrix(ytest.Y, res);


# @param full_neg: dataframe containing all negative responses for that bidder
# @param df_pos_size: amount of data in the positive dataframe
# @param amount_neg_def: how many response_negative dataframes the function will produce
# @param pos_neg_ratio: what the ratio of positive to negative responses will be
# @return a list of negative response dataframes, each considered for one run
def createNegativeResponses(full_neg, pos_df_size, amount_neg_df, pos_neg_ratio):
    all_negatives = [];
    sample_size = math.ceil(pos_df_size * (pos_neg_ratio + 1));
    for count in range(amount_neg_df):
        all_negatives.append(full_neg.sample(sample_size, random_state=count));
    return all_negatives;



# =======================
# DATA CLEANING FUNCTIONS
# =======================


# @param val: a value to be casted to numeric
# @return a  value that has been casted to an integer. Returns 0 if cast was not possible
def tonumeric(val):
	try:
		return int(val)
	except:
		return 0

# @param val: a string value to be categorised
# @return uniffied gatt_wto resulting in either "Yes", "No" or "?"
def unifyYesNo(val):
    switcher = {
        'Ja': 1,
        'Sì': 1,
        'Oui': 1,
        'Nein': 0,
        'Nei': 0,
        'Non': 0,
    }
    return switcher.get(val, 0)

# TODO: Kategorien mit Matthias absprechen
# @param v: the price of a procurement
# @return map prices to 16 categories
def createPriceCategory(val):
    try:
        val = int(val)
    except:
        val = -1
    if val == 0:
        return 0
    if 0 < val <= 100000:
        return 1
    if 100000 < val <= 250000:
        return 2
    if 250000 < val <= 500000:
        return 3
    if 500000 < val <= 750000:
        return 4
    if 750000 < val <= 1000000:
        return 5
    if 1000000 < val <= 2500000:
        return 6
    if 2500000 < val <= 5000000:
        return 7
    if 5000000 < val <= 10000000:
        return 8
    if 10000000 < val <= 25000000:
        return 9
    if 25000000 < val <= 50000000:
        return 10
    if 50000000 < val <= 100000000:
        return 11
    if 100000000 < val <= 200000000:
        return 12
    if 200000000 < val <= 500000000:
        return 13
    if val > 500000000:
        return 14
    else:
        return -1






