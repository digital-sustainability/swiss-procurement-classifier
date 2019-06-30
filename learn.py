import pandas as pd
import numpy as np
import math
import re
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef
from sklearn import tree

from db import connection, engine

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ModelTrainer():

    def __init__(self, select, anbieter, config, attributes=[]):
        self.anbieter = anbieter
        self.select = select
        self.attributes = attributes
        self.config = config

    def run(self):
        self.queryData()
        prepared_positives, prepared_negatives, duplicates = self.prepare_data()

        result = self.trainAllModels(prepared_positives, prepared_negatives)

        result['duplicates'] = duplicates.to_dict()

        return result

    def resetSQLData(self):
        try:
            del self.positives
            del self.negatives
        except:
            pass

    def trainAllModels(self, positives, negatives):

        result = {
            'attributes': self.attributes,
            'anbieter': self.anbieter,
            'timestamp': datetime.now().isoformat()
        }
        samples = self.createSamples(positives, negatives)
        result = {**result, **self.trainAllAlgorithms(samples)}

        return result

    def createSamples(self, positives, negatives):
        negative_sample_size = math.ceil(len(positives) * (self.config['positive_to_negative_ratio'] + 1))
        samples = []
        for runIndex in range(self.config['runs']):
            negative_sample = negatives.sample(negative_sample_size, random_state=runIndex)

            sample = positives.append(negative_sample, ignore_index=True)
            sample.reset_index(drop=True, inplace=True)
            sample.fillna(0, inplace=True)
            sample = shuffle(sample, random_state=runIndex)
            samples.append(sample)
        return samples

    def trainAllAlgorithms(self, samples):
        result = {}
        for algorithm in self.config['enabled_algorithms']:
            if algorithm == 'random_forest':
                n_estimators = self.config[algorithm]['n_estimators']
                max_depth = self.config[algorithm]['max_depth']
                max_features = self.config[algorithm]['max_features']
                min_samples_split = self.config[algorithm]['min_samples_split']
                classifier = lambda randomState: RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    max_features=max_features,
                    min_samples_split=min_samples_split,
                    random_state=randomState,
                    n_jobs=-1
                )
            elif algorithm == 'gradient_boost':
                n_estimators = self.config[algorithm]['n_estimators']
                max_depth = self.config[algorithm]['max_depth']
                max_features = self.config[algorithm]['max_features']
                learning_rate = self.config[algorithm]['learning_rate']
                classifier = lambda randomState: GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    max_features=max_features,
                    learning_rate=learning_rate,
                    random_state=randomState
                )
            elif algorithm == 'decision_tree':
                max_depth = self.config[algorithm]['max_depth']
                max_features = self.config[algorithm]['max_features']
                classifier = lambda randomState: DecisionTreeClassifier(
                    max_depth=max_depth,
                    max_features=max_features
                )
            else:
                raise Exception('enabled algorithm: {} doesn\'t exist.'.format(algorithm))
            result[algorithm] = {}
            x_tests, y_tests = self.trainModel(samples, classifier, algorithm)

            result[algorithm]['metrics'] = self.config[algorithm]
            evaluation_dataframe = pd.concat([self.__getConfusionMatices(y_tests), self.__getAccuracies(y_tests)], axis=1, sort=False)
            result[algorithm]['data'] = evaluation_dataframe.to_dict()
            result[algorithm]['metadata'] = self.__getIterationMetadata(evaluation_dataframe)

        return result

    def trainModel(self, samples, get_classifier, algorithm):
        x_tests = []
        y_tests = []
        for runIndex, sample in enumerate(samples):
            classifier = get_classifier(runIndex)
            train, test = train_test_split(sample, random_state=runIndex)

            if 'skip_cross_val' not in self.config or not self.config['skip_cross_val']:
                # Compute cross validation (5-fold)
                scores = self.__cross_val_score(classifier, train, cv=5)
                print(scores)
                print('Avg. CV Score | {} Run {}: {:.2f}'.format(algorithm, runIndex, round(sum(scores)/len(scores), 4)))

            # Select all attributes
            x_test = test.drop(['Y'], axis=1)
            x_train = train.drop(['Y'], axis=1)
            # Only select the response result attributes
            y_test = test[['Y']].copy()
            y_train = train[['Y']]
            # Create the model
            # Train the model on training sets
            classifier = classifier.fit(x_train, y_train['Y'])

            # print the max_depths of all classifiers in a Random Forest
            if algorithm == 'random_forest':
                print('Random Forest Depts:', [self.dt_max_depth(t.tree_) for t in classifier.estimators_])
            # Create a file displaying the tree
            if 'draw_tree' in self.config and self.config['draw_tree'] and algorithm == 'decision_tree' and runIndex == 0:
                tree.export_graphviz(classifier, out_file='tree.dot', feature_names=x_train.columns)

            # Predict on the test sets
            prediction = classifier.predict(x_test)

            # Add run number to df
            y_test['run'] = runIndex
            x_test['run'] = runIndex
            # add prediction to df
            y_test['prediction'] = prediction
            # add result of run to df
            y_test['correct'] = y_test['prediction'] == y_test['Y']
            # add run to run arrays
            x_tests.append(x_test)
            y_tests.append(y_test)
        return x_tests, y_tests


    def queryData(self):
        if not hasattr(self, 'positives') or not hasattr(self, 'negatives'):
            self.positives = self.__runSql(True)
            self.negatives = self.__runSql(False)
            logger.info('sql done')
        return self.positives, self.negatives

    def __runSql(self, response):
        resp = '='
        if (not response):
            resp = '!='
        query = """SELECT {} from beruecksichtigteanbieter_zuschlag
                JOIN zuschlag ON zuschlag.meldungsnummer = beruecksichtigteanbieter_zuschlag.meldungsnummer
                JOIN anbieter ON beruecksichtigteanbieter_zuschlag.anbieter_id = anbieter.anbieter_id
                JOIN projekt ON zuschlag.projekt_id = projekt.projekt_id
                JOIN auftraggeber ON projekt.auftraggeber_id = auftraggeber.auftraggeber_id
                JOIN ausschreibung ON projekt.projekt_id = ausschreibung.projekt_id
                JOIN cpv_dokument ON cpv_dokument.meldungsnummer = ausschreibung.meldungsnummer
                WHERE anbieter.institution {} "{}"
                ORDER BY ausschreibung.meldungsnummer;
        """.format(self.select, resp, self.anbieter)
        return pd.read_sql(query, engine)

    def prepareUnfilteredRun(self, positive_sample, negative_samples):
        merged_samples_for_names = []
        for negative_sample in negative_samples:
            # Merge positive and negative df into one
            merged_samples_for_names.append(positive_sample.append(negative_sample, ignore_index=True).copy())
        return  merged_samples_for_names


    def __getAccuracies(self, dfys):
        res = pd.DataFrame(columns=['accuracy', 'MCC', 'fn_rate'])
        for dfy in dfys:
            acc = round(accuracy_score(dfy.Y, dfy.prediction), 4)
            # f1 = round(f1_score(dfy.Y, dfy.prediction), 4)
            mcc = matthews_corrcoef(dfy.Y, dfy.prediction)
            matrix = confusion_matrix(dfy.Y, dfy.prediction)
            fnr = round(matrix[1][0] / (matrix[1][1] + matrix[1][0]), 4)
            # add row to end of df, *100 for better % readability
            res.loc[len(res)] = [ acc*100, mcc, fnr*100 ]
        return res

    def __getConfusionMatices(self, dfys):
        res = pd.DataFrame(columns=['tn', 'tp', 'fp', 'fn'])
        for dfy in dfys:
            # ConfusionMatrix legende:
            # [tn, fp]
            # [fn, tp]
            matrix = confusion_matrix(dfy.Y, dfy.prediction)
            res.loc[len(res)] = [ matrix[0][0], matrix[1][1], matrix[0][1], matrix[1][0] ]
        # res.loc['sum'] = res.sum()  # Summarize each column
        return res

    def __getIterationMetadata(self, df):
        res = {}
        res['acc_mean'] = df['accuracy'].mean()
        res['acc_median'] = df['accuracy'].median()
        res['acc_min'] = df['accuracy'].min()
        res['acc_max'] = df['accuracy'].max()
        res['acc_quantile_25'] = df['accuracy'].quantile(q=.25)
        res['acc_quantile_75'] = df['accuracy'].quantile(q=.75)

        res['mcc_mean'] = df['MCC'].mean()
        res['mcc_median'] = df['MCC'].median()
        res['mcc_min'] = df['MCC'].min()
        res['mcc_max'] = df['MCC'].max()
        res['mcc_quantile_25'] = df['MCC'].quantile(q=.25)
        res['mcc_quantile_75'] = df['MCC'].quantile(q=.75)

        res['fn_rate_mean'] = df['fn_rate'].mean()
        res['fn_rate_median'] = df['fn_rate'].median()
        res['fn_rate_min'] = df['fn_rate'].min()
        res['fn_rate_max'] = df['fn_rate'].max()
        res['fn_rate_quantile_25'] = df['fn_rate'].quantile(q=.25)
        res['fn_rate_quantile_75'] = df['fn_rate'].quantile(q=.75)

        res['sample_size_mean'] = (df['fp'] + df['fn'] + df['tn'] + df['tp']).mean()
        return res

    def __cross_val_score(self, clf, sample, cv):

        cross_val_scores = []
        for validation_run_index in range(cv):
            train, test = train_test_split(sample, random_state=validation_run_index)
            # Select all attributes but meldungsnummer
            xtest = test.drop(['Y'], axis=1)
            xtrain = train.drop(['Y'], axis=1)
            # Only select the response result attributes
            ytest = test[['Y']]
            ytrain = train[['Y']]

            clf = clf.fit(xtrain, ytrain['Y'])

            prediction = clf.predict(xtest)

            cross_val_scores.append(accuracy_score(ytest, prediction))
        return cross_val_scores

    def prepare_data(self):

        filter_attributes = ['meldungsnummer'] + self.attributes
        # filter only specified attributes

        positives = self.positives[filter_attributes].copy()
        negatives = self.negatives[filter_attributes].copy()

        positives['Y'] = 1
        negatives['Y'] = 0

        merged = positives.append(negatives, ignore_index=True)

        if hasattr(self, 'cleanData'):
            positives = self.cleanData(positives, self.attributes)
            negatives = self.cleanData(negatives, self.attributes)

        else:
            # positives = self.preprocess_data(positives, self.attributes)
            # negatives = self.preprocess_data(negatives, self.attributes)
            merged, duplicates = self.preprocess_data(merged, self.attributes)


        positives = merged[merged['Y']==1]
        negatives = merged[merged['Y']==0]

        return positives, negatives, duplicates


    def preprocess_data(self, df, filters):
        df = df.copy()
        # drop duplicates before starting to preprocess
        df = df.drop_duplicates()

        if 'ausschreibung_cpv' in filters:
            split = {
                'division': lambda x: math.floor(x/1000000),
                'group': lambda x: math.floor(x/100000),
                'class': lambda x: math.floor(x/10000),
                'category': lambda x: math.floor(x/1000)
            }
            for key, applyFun in split.items():
                df['cpv_' + key ] = df['ausschreibung_cpv'].apply(applyFun)

            tmpdf = {}
            for key in split.keys():
                key = 'cpv_' + key
                tmpdf[key] = df[['meldungsnummer']].join(pd.get_dummies(df[key], prefix=key)).groupby('meldungsnummer').max()

            encoded_df = pd.concat([tmpdf['cpv_'+ key] for key in split.keys()], axis=1)
            df = df.drop(['cpv_' + key for key, fun in split.items()], axis=1)

            df = df.drop(['ausschreibung_cpv'], axis=1)
            df = df.drop_duplicates()

            df = df.join(encoded_df, on='meldungsnummer')


        if 'gatt_wto' in filters:
            df[['gatt_wto']] = df[['gatt_wto']].applymap(ModelTrainer.unifyYesNo)
        if 'anzahl_angebote' in filters:
            df[['anzahl_angebote']] = df[['anzahl_angebote']].applymap(ModelTrainer.tonumeric)
        if 'teilangebote' in filters:
            df[['teilangebote']] = df[['teilangebote']].applymap(ModelTrainer.unifyYesNo)
        if 'lose' in filters:
            df[['lose']] = df[['lose']].applymap(ModelTrainer.unifyYesNoOrInt)
        if 'varianten' in filters:
            df[['varianten']] = df[['varianten']].applymap(ModelTrainer.unifyYesNo)
        if 'auftragsart_art' in filters:
            auftrags_art_df = pd.get_dummies(df['auftragsart_art'], prefix='aftrgsrt', dummy_na=True)
            df = pd.concat([df,auftrags_art_df],axis=1).drop(['auftragsart_art'], axis=1)
        if 'sprache' in filters:
            sprache_df = pd.get_dummies(df['sprache'], prefix='lang', dummy_na=True)
            df = pd.concat([df,sprache_df],axis=1).drop(['sprache'], axis=1)
        if 'auftragsart' in filters:
            auftragsart_df = pd.get_dummies(df['auftragsart'], prefix='auftr', dummy_na=True)
            df = pd.concat([df,auftragsart_df],axis=1).drop(['auftragsart'], axis=1)
        if 'beschaffungsstelle_plz' in filters:
            # plz_df = pd.get_dummies(df['beschaffungsstelle_plz'], prefix='beschaffung_plz', dummy_na=True)
            # df = pd.concat([df,plz_df],axis=1).drop(['beschaffungsstelle_plz'], axis=1)
            df['beschaffungsstelle_plz'] = df['beschaffungsstelle_plz'].apply(ModelTrainer.transformToSingleInt)
            split = {
                'district': lambda x: math.floor(x/1000) if not math.isnan(x) else x,
                'area': lambda x: math.floor(x/100) if not math.isnan(x) else x,
            }
            prefix = 'b_plz_'

            for key, applyFun in split.items():
                df[prefix + key] = df['beschaffungsstelle_plz'].apply(applyFun)

            df.rename(columns={'beschaffungsstelle_plz': prefix + 'ganz'}, inplace=True)

            for key in ['ganz'] + list(split.keys()):
                key = prefix + key
                df = pd.concat([df, pd.get_dummies(df[key], prefix=key, dummy_na=True)], axis=1).drop(key, axis=1)

        df.drop_duplicates(inplace=True)
        if any(df.duplicated(['meldungsnummer'])):
            logger.warning("duplicated meldungsnummer")
            duplicates = df[df.duplicated(['meldungsnummer'])]

        df = df.drop(['meldungsnummer'], axis=1)

        return df, duplicates

    def dt_max_depth(self, tree):
        n_nodes = tree.node_count
        children_left = tree.children_left
        children_right = tree.children_right
        def walk(node_id):
            if (children_left[node_id] != children_right[node_id]):
                left_max = 1 + walk(children_left[node_id])
                right_max = 1 + walk(children_right[node_id])
                return max(left_max, right_max)
            else: # is leaf
                return 1
        root_node_id = 0
        return walk(root_node_id)


    # @param val: a value to be casted to numeric
    # @return a  value that has been casted to an integer. Returns 0 if cast was not possible
    def tonumeric(val):
        try:
            return int(val)
        except:
            return 0

    # @param val: a string value to be categorised
    # @return uniffied gatt_wto resulting in either "Yes", "No" or "?"
    @staticmethod
    def unifyYesNo(val):
        switcher = {
            'Ja': 1,
            'Sì': 1,
            'Oui': 1,
            'YES': 1,
            'Nein': 0,
            'Nei': 0,
            'Non': 0,
            'NO': 0,
        }
        return switcher.get(val, 0)

    @staticmethod
    def unifyYesNoOrInt(val):
        try:
            return int(val)
        except ValueError:
            return ModelTrainer.unifyYesNo(val)

    @staticmethod
    def transformToSingleInt(plz):
        try:
            result = int(plz)

        except ValueError:
            try:
                result = int(re.search(r"\d{4}", plz).group())
            except AttributeError:
                return np.nan

        return result if result >= 1000 and result <= 9999 else np.nan
