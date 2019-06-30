from learn import ModelTrainer
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
        'KIBAG',
        'Egli AG',
    ],
    'IT': [
        'Swisscom',
        'ELCA Informatik AG',
        'Unisys',
    ],
    'Other': [
        'Kummler + Matter AG',
        'Thermo Fisher Scientific (Schweiz) AG',
        'AXA Versicherung AG',
    ],
    'Diverse': [
        'Siemens AG',
        'ABB',
        'Basler & Hofmann West AG',
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


select = (
    "ausschreibung.meldungsnummer, "
    "anbieter.institution as anbieter_institution, "
    "auftraggeber.beschaffungsstelle_plz, "
    "ausschreibung.gatt_wto, "
    "ausschreibung.sprache, "
    "ausschreibung.auftragsart, "
    "ausschreibung.auftragsart_art, "
    "ausschreibung.lose, "
    "ausschreibung.teilangebote, "
    "ausschreibung.varianten, "
    "ausschreibung.bietergemeinschaft, "
    "cpv_dokument.cpv_nummer as ausschreibung_cpv"
)

attributes = ['ausschreibung_cpv', 'auftragsart_art', 'beschaffungsstelle_plz', 'auftragsart', 'gatt_wto','lose','teilangebote', 'varianten','sprache']
#attributes = ['auftragsart_art', 'beschaffungsstelle_plz', 'auftragsart', 'ausschreibung_cpv', 'gatt_wto','teilangebote', 'sprache']
#attributes = ['ausschreibung_cpv', 'auftragsart_art', 'beschaffungsstelle_plz', 'auftragsart', 'gatt_wto','lose','teilangebote', 'varianten','sprache']
# attributes = [
#       [ 'ausschreibung_cpv', 'auftragsart_art' ],
#       [ 'ausschreibung_cpv', 'beschaffungsstelle_plz' ],
#       [ 'ausschreibung_cpv', 'auftragsart' ],
#       [ 'ausschreibung_cpv', 'gatt_wto' ],
#       [ 'ausschreibung_cpv', 'lose' ],
#       [ 'ausschreibung_cpv', 'teilangebote' ],
#       [ 'ausschreibung_cpv', 'varianten' ],
#       [ 'ausschreibung_cpv', 'sprache' ]
# ]

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
        'min_samples_split': 4
    },
    'decision_tree': {
        'max_depth': 30,
        'max_features': 'sqrt',
        'min_samples_split': 4
    },
    'gradient_boost': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 30,
        'min_samples_split': 4,
        'max_features': 'sqrt'
    }
}


class IterationRunner():

    def __init__(self, anbieter_config, select, attributes, config):
        self.anbieter_config = anbieter_config
        self.select = select
        self.attributes = attributes
        self.config = config
        self.trainer = ModelTrainer(select, '', config, attributes)
        self.collection = Collection()

    def run(self):
        for label, anbieters in self.anbieter_config.items():
            logger.info(label)
            for anbieter in anbieters:
                for attr_id in range(len(self.attributes)):
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

    def runAttributesList(self):
        for label, anbieters in self.anbieter_config.items():
            logger.info(label)
            for anbieter in anbieters:
                for att_list in self.attributes:
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

runner = IterationRunner(anbieter_config, select, attributes, config)

if __name__ == '__main__':
    # runner.collection.import_file('dbs/auto.json')
    runner.run()
    runner.runAttributesEachOne()
    runner.runAttributesList()
    # label, anbieters = next(iter(runner.anbieter_config.items()))
    # print(label)
