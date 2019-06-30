import json
import pandas as pd
import warnings

class Collection():

    algorithms = ['gradient_boost', 'decision_tree', 'random_forest']

    def __init__(self):
        self.list = []


    def append(self, item):
        self.list.append(item)

    def __iter__(self):
        return iter(self.list)

    def get_all_as_df(self, algorithm):
        try:
            tmp = []
            for iteration in self.list:
                tmp.append(iteration[algorithm]['metadata'])
            return pd.DataFrame(tmp, index=[iteration['anbieter'] for iteration in self.list])
        except:
            warnings.warn('Select an algorithm: "random_forest", "gradient_boost" or "decision_tree"')

    def df_row_per_algorithm(self):
        tmp = []
        for iteration in self.list:
            for algorithm in self.algorithms:
                output = iteration[algorithm]['metadata']
                evaluation_dataframe = pd.DataFrame.from_dict(iteration[algorithm]['data'])
                # missing metrics
                output['acc_std'] = evaluation_dataframe['accuracy'].std()
                evaluation_dataframe['MCC'] = evaluation_dataframe['MCC']*100
                output['mcc_std'] = evaluation_dataframe['MCC'].std()
                output['fn_std'] = evaluation_dataframe['fn_rate'].std()

                output['anbieter'] = iteration['anbieter']
                output['label'] = iteration['label']
                output['algorithm'] = algorithm
                output['attributes'] = ",".join(iteration['attributes'])
                tmp.append(output)
        return pd.DataFrame(tmp)

    def to_json(self, **kwargs):
        return json.dumps(self.list, **kwargs)

    def to_file(self, filename):
        with open(filename, 'w') as fp:
            json.dump(self.list, fp, indent=4, sort_keys=True)

    def import_file(self, filename, force=False):
        if len(self.list) and not force:
            warnings.warn("Loaded Collection, pls add force=True")
        else:
            with open(filename, 'r') as fp:
                self.list = json.load(fp)
