import pandas as pd


class FeatureDictionary(object):
    def __init__(self, dfTrain=None, dfTest=None, numeric_cols=[], ignore_cols=[]):
        self.dfTrain = dfTrain
        self.dfTest = dfTest
        self.numeric_cols = numeric_cols
        self.ignore_cols = ignore_cols
        self.get_feat_dict()

    def get_feat_dict(self):

        dfTrain = self.dfTrain
        dfTest = self.dfTest
        df = pd.concat([dfTrain, dfTest], sort=False)

        self.feat_dict = {}
        tc = 0
        for col in df.columns:
            if col in self.ignore_cols:
                continue
            if col in self.numeric_cols:    #TODO:这个numeric_cols是用来做什么的？
                self.feat_dict[col] = tc
                tc += 1

            else:
                us = df[col].unique()
                self.feat_dict[col] = dict(zip(us, range(tc, len(us) + tc)))
                tc += len(us)

        self.feat_dim = tc

class DataParser(object):
    def __init__(self, feat_dict):
        self.feat_dict = feat_dict

    def parse(self, df=None):

        dfi = df.copy()
        y = dfi['review_ratting'].values.tolist()
        # TODO: 删掉了‘id’
        dfi.drop(['review_ratting'], axis=1, inplace=True)

        # dfi for feature index
        # dfv for feature value which can be either binary (1/0) or float (e.g., 10.24)
        dfv = dfi.copy()
        for col in dfi.columns:
            if col in self.feat_dict.ignore_cols:
                dfi.drop(col, axis=1, inplace=True)
                dfv.drop(col, axis=1, inplace=True)
                continue
            if col in self.feat_dict.numeric_cols:
                dfi[col] = self.feat_dict.feat_dict[col]
            else:
                dfi[col] = dfi[col].map(self.feat_dict.feat_dict[col])
                dfv[col] = 1.

        xi = dfi.values.tolist()
        xv = dfv.values.tolist()

        return xi, xv, y