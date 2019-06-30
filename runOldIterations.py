from train import ModelTrainer
from collection import Collection
import pandas as pd

import logging
import traceback
import os

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# === THESIS ===

anbieter_config = {
    'Construction': [
        'Alpiq AG',
        'Swisscom',
        'Kummler + Matter AG',
        'Siemens AG'
    ],
    'IT': [
        'G. Baumgartner AG',
        'ELCA Informatik AG',
        'Thermo Fisher Scientific (Schweiz) AG',
        'Arnold AG',
    ],
    'Other': [
        'Riget AG',
        'isolutions AG',
        'CSI Consulting AG',
        'Aebi & Co. AG Maschinenfabrik',
    ],
    'Divers': [
        'DB Schenker AG',
        'IT-Logix AG',
        'AVS Syteme AG',
        'Sajet SA'
    ]
}



# === TESTING ===

#anbieter = 'Marti AG' #456
#anbieter = 'Axpo AG' #40
#anbieter = 'Hewlett-Packard' #90
#anbieter = 'BG Ingénieurs Conseils' SA #116
#anbieter = 'Pricewaterhousecoopers' #42
#anbieter = 'Helbling Beratung + Bauplanung AG' #20
#anbieter = 'Ofrex SA' #52
#anbieter = 'PENTAG Informatik AG' #10
#anbieter = 'Wicki Forst AG' #12
#anbieter = 'T-Systems Schweiz' #18
#anbieter = 'Bafilco AG' #20
#anbieter = '4Video-Production GmbH' #3
#anbieter = 'Widmer Ingenieure AG' #6
#anbieter = 'hmb partners AG' #2
#anbieter = 'Planmeca' #4
#anbieter = 'K & M Installationen AG' #4


select_anbieter = (
    "anbieter.anbieter_id, "
    "anbieter.institution as anbieter_institution, "
    "cpv_dokument.cpv_nummer as anbieter_cpv, "
    "ausschreibung.meldungsnummer"
)
# anbieter_CPV are all the CPVs the Anbieter ever won a procurement for. So all the CPVs they are interested in. 
select_ausschreibung = (
    "anbieter.anbieter_id, "
    "auftraggeber.institution as beschaffungsstelle_institution, "
    "auftraggeber.beschaffungsstelle_plz, "
    "ausschreibung.gatt_wto, "
    "ausschreibung.sprache, "
    "ausschreibung.auftragsart_art, "
    "ausschreibung.lose, "
    "ausschreibung.teilangebote, "
    "ausschreibung.varianten, "
    "ausschreibung.projekt_id, "
   # "ausschreibung.titel, "
    "ausschreibung.bietergemeinschaft, "
    "cpv_dokument.cpv_nummer as ausschreibung_cpv, "
    "ausschreibung.meldungsnummer as meldungsnummer2"
)

attributes = ['ausschreibung_cpv', 'auftragsart_art','beschaffungsstelle_plz','gatt_wto','lose','teilangebote', 'varianten','sprache']
# attributes = ['auftragsart_art']

config = {
    # ratio that the positive and negative responses have to each other
    'positive_to_negative_ratio': 0.5,
    # Percentage of training set that is used for testing (Recommendation of at least 25%)
    'test_size': 0.25,
    'runs': 100,
    #'enabled_algorithms': ['random_forest'],
    'enabled_algorithms': ['random_forest', 'decision_tree', 'gradient_boost'],
    'random_forest': {
        # Tune Random Forest Parameter
        'n_estimators': 100,
        'max_features': 'sqrt',
        'max_depth': None,
        'min_samples_split': 2
    },
    'decision_tree': {
        'max_depth': 15,
        'max_features': 'sqrt'
    },
    'gradient_boost': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 15,
        'max_features': 'sqrt'
    }
}

# Prepare Attributes
def cleanData(df, filters):
#    if 'beschaffungsstelle_plz' in filters:
#        df[['beschaffungsstelle_plz']] = df[['beschaffungsstelle_plz']].applymap(ModelTrainer.tonumeric)
    if 'gatt_wto' in filters:
        df[['gatt_wto']] = df[['gatt_wto']].applymap(ModelTrainer.unifyYesNo)
    if 'anzahl_angebote' in filters:
        df[['anzahl_angebote']] = df[['anzahl_angebote']].applymap(ModelTrainer.tonumeric)
    if 'teilangebote' in filters:
        df[['teilangebote']] = df[['teilangebote']].applymap(ModelTrainer.unifyYesNo)
    if 'lose' in filters:
        df[['lose']] = df[['lose']].applymap(ModelTrainer.unifyYesNo)
    if 'varianten' in filters:
        df[['varianten']] = df[['varianten']].applymap(ModelTrainer.unifyYesNo)
    if 'auftragsart_art' in filters:
        auftrags_art_df = pd.get_dummies(df['auftragsart_art'], prefix='aftrgsrt',dummy_na=True)
        df = pd.concat([df,auftrags_art_df],axis=1).drop(['auftragsart_art'],axis=1)
    if 'sprache' in filters:
        sprache_df = pd.get_dummies(df['sprache'], prefix='lang',dummy_na=True)
        df = pd.concat([df,sprache_df],axis=1).drop(['sprache'],axis=1)
    if 'auftragsart' in filters:
        auftragsart_df = pd.get_dummies(df['auftragsart'], prefix='auftr',dummy_na=True)
        df = pd.concat([df,auftragsart_df],axis=1).drop(['auftragsart'],axis=1)
    if 'beschaffungsstelle_plz' in filters:
        plz_df = pd.get_dummies(df['beschaffungsstelle_plz'], prefix='beschaffung_plz',dummy_na=True)
        df = pd.concat([df,plz_df],axis=1).drop(['beschaffungsstelle_plz'],axis=1)
    return df

class IterationRunner():

    def __init__(self, anbieter_config, select_anbieter, select_ausschreibung, attributes, config, cleanData):
        self.anbieter_config = anbieter_config
        self.select_anbieter = select_anbieter
        self.select_ausschreibung = select_ausschreibung
        self.attributes = attributes
        self.config = config
        self.cleanData = cleanData
        self.trainer = ModelTrainer(select_anbieter, select_ausschreibung, '', config, cleanData, attributes)
        self.collection = Collection()

    def run(self):
        for label, anbieters in self.anbieter_config.items():
            logger.info(label)
            for anbieter in anbieters:
                for attr_id in range(len(self.attributes)-1):
                    att_list = self.attributes[:attr_id+1]
                    self.singleRun(anbieter, att_list, label)
                self.trainer.resetSQLData()

    def runAttributesEachOne(self):
        for label, anbieters in self.anbieter_config.items():
            logger.info(label)
            for anbieter in anbieters:
                for attr in self.attributes:
                    att_list = [attr]
                    self.singleRun(anbieter, att_list, label)
                self.trainer.resetSQLData()


    def runSimpleAttributeList(self):
        for label, anbieters in self.anbieter_config.items():
            logger.info(label)
            for anbieter in anbieters:
                self.singleRun(anbieter, self.attributes, label)
                self.trainer.resetSQLData()

    def singleRun(self, anbieter, att_list, label):
        logger.info('label: {}, anbieter: {}, attributes: {}'.format(label, anbieter, att_list))
        try:
            self.trainer.attributes = att_list
            self.trainer.anbieter = anbieter
            output = self.trainer.run()
            output['label'] = label
            self.collection.append(output)
            filename = os.getenv('DB_FILE', 'dbs/auto.json')
            self.collection.to_file(filename)
        except Exception as e:
            traceback.print_exc()
            print(e)
        print('one it done')

runner = IterationRunner(anbieter_config, select_anbieter, select_ausschreibung, attributes, config, cleanData)

if __name__ == '__main__':
    # runner.collection.import_file('dbs/auto.json')
    runner.run()
    runner.runAttributesEachOne()
    # label, anbieters = next(iter(runner.anbieter_config.items()))
    # print(label)
