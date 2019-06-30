import pandas as pd
import math
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef

from db import connection, engine

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ModelTrainer():

    def __init__(self, select_anbieter, select_ausschreibung, anbieter, config, cleanData, attributes=[]):
        self.anbieter = anbieter
        self.select_anbieter = select_anbieter
        self.select_ausschreibung = select_ausschreibung
        self.attributes = attributes
        self.config = config
        self.cleanData = cleanData

    def run(self):
        positive_sample, negative_samples = self.createSamples()

        positive_and_negative_samples = self.prepareForRun(
            positive_sample,
            negative_samples
        )

        # most certainly used to resolve the naming functions like getFalseProjectTitle
        merged_samples_for_names = self.prepareUnfilteredRun(
            positive_sample,
            negative_samples
        )

        result = self.trainSpecifiedModels(positive_and_negative_samples)

        return result
        # xTests, yTests = self.trainModel(positive_and_negative_samples)

    def resetSQLData(self):
        try:
            del self.positives
            del self.negatives
        except:
            pass

    def createSamples(self):
        if not hasattr(self, 'positives') or not hasattr(self, 'negatives'):
            self.queryData()
        negative_samples = []
        negative_sample_size = math.ceil(len(self.positives) * (self.config['positive_to_negative_ratio'] + 1))
        for count in range(self.config['runs']):
            negative_samples.append(self.negatives.sample(negative_sample_size, random_state=count))

        self.positives['Y'] = 1
        for negative_sample in negative_samples:
            negative_sample['Y']=0
        return (self.positives, negative_samples)

    def queryData(self):
        self.positives = self.__runSql(True)
        self.negatives = self.__runSql(False)
        logger.info('sql done')
        return self.positives, self.negatives

    def __runSql(self, response):
        resp = '='
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
        """.format(self.select_anbieter, resp, self.anbieter, self.select_ausschreibung, resp, self.anbieter)
        return pd.read_sql(query, engine)

    def prepareForRun(self, positive_sample, negative_samples):
        # What attributes the model will be trained by
        filters = ['Y', 'projekt_id'] + self.attributes
        positive_and_negative_samples = []
        for negative_sample in negative_samples:
        # Merge positive and negative df into one, only use selected attributes
            merged_samples = positive_sample.append(negative_sample, ignore_index=True)[filters].copy()
        # Clean the data of all selected attributes
            cleaned_merged_samples = self.cleanData(merged_samples, self.attributes)
            positive_and_negative_samples.append(cleaned_merged_samples)
        return positive_and_negative_samples

    def prepareUnfilteredRun(self, positive_sample, negative_samples):
        merged_samples_for_names = []
        for negative_sample in negative_samples:
            # Merge positive and negative df into one
            merged_samples_for_names.append(positive_sample.append(negative_sample, ignore_index=True).copy())
        return  merged_samples_for_names

    def trainSpecifiedModels(self, positive_and_negative_samples):
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
            xTests, yTests = self.trainModel(positive_and_negative_samples, classifier, algorithm)
            result['attributes'] = self.attributes
            result['anbieter'] = self.anbieter
            result['timestamp'] = datetime.now().isoformat()
            #result[algorithm]['xTests'] = xTests
            #result[algorithm]['yTests'] = yTests
            result[algorithm]['metrics'] = self.config[algorithm]
            evaluation_dataframe =pd.concat([self.__getConfusionMatices(yTests), self.__getAccuracies(yTests)], axis=1, sort=False)
            result[algorithm]['data'] = evaluation_dataframe.to_dict()
            result[algorithm]['metadata'] = self.__getIterationMetadata(evaluation_dataframe)
        return result

    def trainModel(self, positive_and_negative_samples, classifier, algorithm):
        xTests = []
        yTests = []
        for idx, df in enumerate(positive_and_negative_samples): # enum to get index
            x_and_y_test, x_and_y_train = self.unique_train_and_test_split(df, random_state=idx)
            # Select all attributes
            xtest = x_and_y_test.drop(['Y'], axis=1)
            xtrain = x_and_y_train.drop(['Y'], axis=1)
            # Only select the response result attributes
            ytest = x_and_y_test['Y']
            ytrain = x_and_y_train['Y']
            # Create the model
            clf = classifier(randomState=idx)
            # Compute cross validation (5-fold)
            scores = self.__cross_val_score(clf, xtest, ytest, cv=5)
            print(scores)
            print('Avg. CV Score | {} Run {}: {:.2f}'.format(algorithm, idx, round(sum(scores)/len(scores), 4)))

            xtest = xtest.drop(['projekt_id'], axis=1)
            xtrain = xtrain.drop(['projekt_id'], axis=1)
            # Train the model on training sets
            clf = clf.fit(xtrain, ytrain)
            # Predict on the test sets
            prediction = clf.predict(xtest)
            # Convert pandas.series to data frame
            df_ytest = ytest.to_frame()
            # Add run number to df
            df_ytest['run'] = idx
            xtest['run'] = idx
            # add prediction to df
            df_ytest['prediction']= prediction
            # add result of run to df
            df_ytest['correct'] = df_ytest['prediction']==df_ytest['Y']
            # add run to run arrays
            xTests.append(xtest)
            yTests.append(df_ytest)
        return xTests, yTests

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

    def __cross_val_score(self, clf, x_values, y_values, cv):
        x_and_y_values = pd.concat([y_values, x_values], axis=1)

        cross_val_scores = []
        for validation_run_index in range(cv):
            x_and_y_test, x_and_y_train = self.unique_train_and_test_split(x_and_y_values, random_state=validation_run_index)
            # Select all attributes but meldungsnummer
            xtest = x_and_y_test.drop(['projekt_id', 'Y'], axis=1)
            xtrain = x_and_y_train.drop(['projekt_id', 'Y'], axis=1)
            # Only select the response result attributes
            ytest = x_and_y_test['Y']
            ytrain = x_and_y_train['Y']

            clf = clf.fit(xtrain, ytrain)

            prediction = clf.predict(xtest)

            cross_val_scores.append(accuracy_score(ytest, prediction))
        return cross_val_scores

    def unique_train_and_test_split(self, df, random_state):
        run = shuffle(df, random_state=random_state)  # run index as random state
        # Get each runs unique meldungsnummer
        unique_mn = run.projekt_id.unique()
        # Split the meldungsnummer between test and trainings set so there will be no bias in test set
        x_unique_test, x_unique_train = train_test_split(unique_mn, test_size=self.config['test_size'], random_state=random_state)
        # Add the remaining attributes to meldungsnummer
        x_and_y_test = run[run['projekt_id'].isin(x_unique_test)].copy()
        x_and_y_train = run[run['projekt_id'].isin(x_unique_train)].copy()
        return x_and_y_test, x_and_y_train


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



