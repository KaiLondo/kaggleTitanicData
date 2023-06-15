import pandas as pd
import numpy as np
from scipy.stats import zscore

class DataFrameAnalyzer:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.numeric_dataframe = self.dataframe.select_dtypes(exclude='object')

    
    def generate_dataframe_summary(self):
        """
        Generates a summary of the dataframe.
        """
        summary = pd.DataFrame({
            'features': self.dataframe.columns,
            'dataType': self.dataframe.dtypes,
            'unqValCount': self.dataframe.nunique(),
            'nullsCount': self.dataframe.isna().sum()
        })

        summary['nullsPct'] = (summary['nullsCount'] / self.dataframe.shape[0]).round(4) * 100
        summary['Unique values(All if < 10)'] = summary['features'].apply(lambda col: self.dataframe[col].unique()[:10])

        return summary.sort_values(by=["dataType", 'unqValCount'], ascending=False).reset_index(drop=True)


def encode_features(self, features, method='onehot'):
        """
        Encodes specified features using either one-hot or label encoding.

        Parameters:
        features: A list of feature names to encode.
        method: The encoding method to use. Either 'onehot' or 'label'. Default is 'onehot'.
        """
        encoded_df = self.dataframe.copy()

        if method == 'onehot':
            encoder = OneHotEncoder()
            for feature in features:
                encoded_feature = pd.DataFrame(encoder.fit_transform(encoded_df[[feature]]).toarray())
                encoded_feature.columns = encoder.get_feature_names([feature])
                encoded_df = pd.concat([encoded_df, encoded_feature], axis=1).drop([feature], axis=1)
        elif method == 'label':
            encoder = LabelEncoder()
            for feature in features:
                encoded_df[feature] = encoder.fit_transform(encoded_df[feature])
        else:
            raise ValueError("Method must be either 'onehot' or 'label'.")

        return encoded_df