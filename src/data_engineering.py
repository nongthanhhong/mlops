import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from utils import AppPath

import matplotlib.pyplot as plt
from sklearn.utils import resample
from problem_config import ProblemConfig, ProblemConst, get_prob_config, load_feature_configs_dict

def build_category_features(data, categorical_cols=None):
        if categorical_cols is None:
            categorical_cols = []
        category_index = {}
        if len(categorical_cols) == 0:
            return data, category_index

        df = data.copy()
        # process category features
        for col in categorical_cols:
            df[col] = df[col].astype("category")
            category_index[col] = df[col].cat.categories
            df[col] = df[col].cat.codes
        return df, category_index

def remove_files_in_folder(folder_path):

    """
    Remove all files in a folder.

    Parameters:
    -----------
    folder_path : str
        The path to the folder containing the files to be removed.
    """
    for file_name in os.listdir(folder_path):
        if file_name == "category_index.pickle":
            continue
        file_path = os.path.join(folder_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
            pass


class SubValues:
  '''
  This class define sub values for create new features
  '''
  #avarge amount of:
  avg_item: dict
  avg_hour: dict
  avg_type: dict
  avg_hour_item: dict

  #proportion/percent of:
  item_job: dict
  item_hour: dict
  item_type: dict
  job_type: dict
  job_hour: dict
  hour_type: dict

  #proportion to label
  hour_fraud: dict
  job_fraud: dict
  type_fraud: dict

class Calculator:
  '''
  This class define functions to calculate sub values
  '''

  #avarge amount of:
  def avg_1(self, data, column, amount):
    '''
    Calculate avarage amount for each unique object in 1 column
    example: 
    A=[1,2,2,1,3,3] 
    Amount=[10,3,4,8,5,6]

       A  avg  
    0  1  9   
    1  2  3.5   
    2  3  5.5

    '''

    result = data.groupby(column)[amount].mean()
    return result.to_dict()

  def avg_2(self,data, column_1, column_2, amount):
    '''
    Calculate avarage amount for each pairs of 2 columns
    example: 
    A=     [1,2,2,1]
    B=     [4,5,4,5] 
    Amount=[10,3,4,8]

       A  B    avg
    0  1  4    10
    1  1  5     8
    2  2  4     4
    3  2  5     3
    '''

    # Group by the item and hour columns and calculate the mean of the amount column
    result = data.groupby([column_1, column_2])[amount].mean()

    # Convert the resulting Series into a DataFrame
    result_df = result.reset_index()

    # Combine the item and hour columns into a single column
    result_df['A_B'] = result_df[column_1].astype(str) + '_' + result_df[column_2].astype(str)

    # Drop the original item and hour columns
    result_df = result_df.drop([column_1, column_2], axis=1)

    return result_df.set_index('A_B')[amount].to_dict()

  #proportion/percent of:
  def percent_1(self,df, column_1, cloumn_2):

    #calculate percent of pair of two columns

    count_pair = df.groupby([column_1, cloumn_2]).size().reset_index(name='count')

    # Combine columns A and B into a single column
    count_pair['A_B'] = count_pair[column_1].astype(str) + '_' + count_pair[cloumn_2].astype(str)

    # Get the count of each value in column B
    count = df[cloumn_2].value_counts().reset_index()
    count.columns = ['cloumn_2', 'count']

    # Merge the dataframes on column B
    merged_df = pd.merge(count_pair, count, left_on=cloumn_2, right_on='cloumn_2')

    # Calculate the percentage
    merged_df['percentage'] = merged_df['count_x'] / merged_df['count_y'] * 100

    # Select only the columns A_B and percentage
    result = merged_df[['A_B', 'percentage']]

    return result.set_index('A_B')['percentage'].to_dict()

  def percent_2(self, df, column_1, label):

    # Calculate percentage of label 1 for each object in column
    result = df.groupby(column_1)[label].mean().reset_index()
    result.columns = [column_1, 'percentage']

    # Convert to dictionary
    result_dict = result.set_index(column_1)['percentage'].to_dict()

    return result_dict


def calculate_sub_values(extractor):

  data = extractor.data
  path_file = extractor.path_file

  sub_values = SubValues()
  calculator = Calculator()

  job = extractor.job
  item = extractor.item
  hour = extractor.hour
  amount = extractor.amount
  type_trans = extractor.type_trans
  label = extractor.label

  #avarge amount of:
  sub_values.avg_item = calculator.avg_1(data, item, amount)
  sub_values.avg_hour = calculator.avg_1(data, hour, amount)
  sub_values.avg_type = calculator.avg_1(data, type_trans, amount)
  sub_values.avg_hour_item = calculator.avg_2(data, hour, item, amount)

  #proportion/percent of:
  sub_values.item_job = calculator.percent_1(data, item, job)
  sub_values.item_hour = calculator.percent_1(data, hour, item)
  sub_values.item_type = calculator.percent_1(data, type_trans, item)
  sub_values.job_type = calculator.percent_1(data, type_trans, job)
  sub_values.job_hour = calculator.percent_1(data, hour, job)
  sub_values.hour_type = calculator.percent_1(data, type_trans, hour)

  sub_values.hour_fraud = calculator.percent_2(data, hour, label)
  sub_values.job_fraud = calculator.percent_2(data, job, label)
  sub_values.type_fraud = calculator.percent_2(data, type_trans, label)


  # Save the instance to a file
  with open(path_file, 'wb') as f:
      pickle.dump(sub_values, f)

  return sub_values


def load_sub_values(path_file):

  # Load the instance from the file
  with open(path_file, 'rb') as f:
      sub_values = pickle.load(f)

  return sub_values



def get_sub_values(extractor):

  if os.path.isfile(extractor.path_file):
    sub_values_dicts = load_sub_values(extractor.path_file)
  else:
    sub_values_dicts = calculate_sub_values(extractor)

  return sub_values_dicts


class FeatureExtractor:
  '''
  Extract new features for dataset
  Input: raw data
  Output: new data (raw data + new extracted features)
  '''
  def __init__(self, df, path_file):
    if df is not None:
        self.data = df.copy()
    self.path_file = path_file # path to saved sub value file
    self.job = 'feature1'
    self.item = 'feature2'
    self.hour = 'feature11'
    self.amount = 'feature3'
    self.type_trans = 'feature13'

    self.user_lat = "feature5"
    self.merchant_lat = "feature8"
    self.user_long = "feature6"
    self.merchant_long = "feature9"

    self.label = 'label'
    self.sub_values_dicts = get_sub_values(self)

  def distance_feature(self, row):

    # distance = sqrt((user_lat - merchant_lat)**2 + (user_long - merchant_long)**2)
    return np.sqrt((row[self.user_lat] - row[self.merchant_lat])**2 + (row[self.user_long] - row[self.merchant_long])**2)

  def avg_item_feature(self, row):
    return abs(row[self.amount] - self.sub_values_dicts.avg_item[row[self.item]])

  def avg_hour_feature(self, row):
    return abs(row[self.amount] - self.sub_values_dicts.avg_hour[row[self.hour]])


  def avg_hour_item_feature(self, row):
    
    query = str(int(row[self.hour])) + '_' + str(int(row[self.item]))
    return abs(row[self.amount] -  self.sub_values_dicts.avg_hour_item[query])

  def percent_item_job_feature(self, row):

    query = str(int(row[self.item])) + '_' + str(int(row[self.job]))
    if '-1' in query:
        return None
    return self.sub_values_dicts.item_job[query]

  def percent_item_hour_feature(self, row):

    query = str(int(row[self.hour])) + '_' + str(int(row[self.item]))
    return self.sub_values_dicts.item_hour[query]

  def percent_job_hour_feature(self, row):

    query = str(int(row[self.hour])) + '_' + str(int(row[self.job]))
    if '-1' in query:
        return None
    return self.sub_values_dicts.job_hour[query]

  def percent_hour_fraud_feature(self, row):

    return self.sub_values_dicts.hour_fraud[row[self.hour]]

  def percent_job_fraud_feature(self, row):

    if '-1' in str(row[self.job]):
        return None
    return self.sub_values_dicts.job_fraud[row[self.job]]

  def create_new_feature(self, raw_data):

    data = raw_data

    data['distance'] = data.apply(self.distance_feature, axis=1)

    data['avg_item'] = data.apply(self.avg_item_feature, axis=1)

    data['avg_hour'] = data.apply(self.avg_hour_feature, axis=1)

    data['avg_hour_item'] = data.apply(self.avg_hour_item_feature, axis=1)

    data['percent_item_job'] = data.apply(self.percent_item_job_feature, axis=1)

    data['percent_item_hour'] = data.apply(self.percent_item_hour_feature, axis=1)

    data['percent_job_hour'] = data.apply(self.percent_job_hour_feature, axis=1)

    data['percent_hour_fraud'] = data.apply(self.percent_hour_fraud_feature, axis=1)

    data['percent_job_fraud'] = data.apply(self.percent_job_fraud_feature, axis=1)

    self.new_data = data
    return self.new_data
  
  def load_new_feature(self, raw_data):

    data = raw_data

    # data['distance'] = data.apply(self.distance_feature, axis=1)

    data['avg_item'] = data.apply(self.avg_item_feature, axis=1)

    data['avg_hour'] = data.apply(self.avg_hour_feature, axis=1)

    data['avg_hour_item'] = data.apply(self.avg_hour_item_feature, axis=1)

    # data['percent_item_job'] = data.apply(self.percent_item_job_feature, axis=1)

    data['percent_item_hour'] = data.apply(self.percent_item_hour_feature, axis=1)

    data['percent_job_hour'] = data.apply(self.percent_job_hour_feature, axis=1)

    data['percent_hour_fraud'] = data.apply(self.percent_hour_fraud_feature, axis=1)

    data['percent_job_fraud'] = data.apply(self.percent_job_fraud_feature, axis=1)

    
    return data


class DataAnalyzer:
    """
    A class for performing exploratory data analysis on tabular data for machine learning.
    """
    def __init__(self, prob_config: ProblemConfig):

        """
        Initializes the DataAnalyzer object with the provided data and target column.
        
        Args:
        data (pandas.DataFrame): The DataFrame containing the data to be analyzed.
        target_col (str): The name of the column containing the target variable.
        """

        
        
        self.raw_path = AppPath.RAW_DATA_DIR / f"{prob_config.phase_id}" / f"{prob_config.prob_id}" 
        self.eda_path = AppPath.EDA_DATA_DIR / f"{prob_config.phase_id}" / f"{prob_config.prob_id}"

        feature_configs = load_feature_configs_dict(self.raw_path / "features_config.json")
        prob_config.target_col = feature_configs.get("target_column")
        prob_config.categorical_cols = feature_configs.get("category_columns")
        prob_config.numerical_cols = feature_configs.get("numeric_columns")
        prob_config.ml_type = feature_configs.get("ml_type")

        self.prob_config = prob_config
        self.data = None
        self.target_col = None
        self.org = None
        self.dtype = None

    def load_data(self):
        # Load data from path
        #save category_index.pickle file for predictor process  transform input

        training_data = pd.read_parquet(self.prob_config.raw_data_path)
        self.data, category_index = build_category_features(
            training_data, self.prob_config.categorical_cols
        )

        dtype = self.data.dtypes.to_frame('dtypes').reset_index().set_index('index')['dtypes'].astype(str).to_dict()
        
        with open(self.prob_config.train_data_path/'types.json', 'w') as f:
            json.dump(dtype, f)

        self.org = training_data
        if not os.path.exists(self.prob_config.train_data_path):
            # Create the new folder
            os.mkdir(self.prob_config.train_data_path)

        with open(self.prob_config.category_index_path, "wb") as f:
            pickle.dump(category_index, f)

        self.target_col = self.prob_config.target_col
    
    def preprocess_data(self, target_col = None, fill_value='mean', scaler=None, method='z-score', threshold=3):

        df = self.data
        dtype = df.dtypes.to_frame('dtypes').reset_index().set_index('index')['dtypes'].astype(str).to_dict()
        data = df.copy()

        # Fill missing values
        if fill_value == 'mean':
            data = data.fillna(data.mean())
        elif fill_value == 'median':
            data = data.fillna(data.median())
        elif isinstance(fill_value, (int, float)):
            data = data.fillna(fill_value)
        else:
            raise ValueError(f"Invalid fill value '{fill_value}'")

        # Remove duplicated rows
        data = data.drop_duplicates()

        # Scale the data
        if scaler is not None:
            if scaler == 'standard':
                data = (data - data.mean()) / data.std()
            elif scaler == 'minmax':
                data = (data - data.min()) / (data.max() - data.min())
            else:
                raise ValueError(f"Invalid scaler '{scaler}'")

        # Drop rows with the wrong format in each column
        for column in data.columns:
            data[column] = pd.to_numeric(data[column], errors='coerce')
            data.dropna(inplace=True)
            data[column] = data[column].astype(dtype[column])


        return data 
        
        if target_col != None: 
            feature_data = data.drop([target_col], axis=1).copy()
        else:  feature_data = data.copy()

        if method == 'z-score':
            z_scores = stats.zscore(feature_data.select_dtypes(include=np.number))
            good_data = feature_data[(abs(z_scores) < threshold).all(axis=1)]

        elif method == 'iqr':
            q1 = feature_data.quantile(0.25)
            q3 = feature_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            good_data = feature_data[~(( feature_data < lower_bound) | ( feature_data > upper_bound)).any(axis=1)]
        else:
            raise ValueError('Invalid method specified. Options are "z-score" and "iqr".')
        
        if target_col != None:
                good_data = good_data.assign(label = data[target_col])

        data = good_data.copy()
        return data.reset_index()

    def feature_selection(self, corr_threshold=0.5, show_chart=False):
        """
        Evaluates the correlation between each feature, and between each feature and the target variable,
        and returns only the features with good correlation.
        
        Args:
        corr_threshold (float): The minimum correlation threshold for a feature to be considered good.
        
        Returns:
        pandas.DataFrame: The subset of the original DataFrame containing only features with good correlation.
        """
        
        # Compute pairwise correlation between features
        corr_matrix = self.data.drop([self.target_col], axis=1).corr()
        
        if show_chart:
            # Plot heatmap of correlation matrix
            plt.figure(figsize=(10, 10))
            sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, square=True)
            plt.title('Pairwise Correlation Between Features')
            plt.show()
        
        # print(self.data.describe())
        # Compute correlation between each feature and the target variable

        corr_with_target = self.data.corr()[self.target_col].sort_values(ascending=False)
        
        print('Correlation Between Each Feature and the Target Variable:\n')
        print(corr_with_target)
        
        if show_chart:
            # Plot bar chart of correlation between each feature and the target variable
            plt.figure(figsize=(10, 5))
            sns.barplot(x=corr_with_target.index, y=corr_with_target.values, palette='coolwarm')
            plt.title('Correlation Between Each Feature and the Target Variable')
            plt.xticks(rotation=90)
            plt.show()
            
        if corr_threshold > np.mean(abs(corr_with_target)):
            corr_threshold = np.mean(abs(corr_with_target))/2

        # Filter out features with low correlation
        good_features = corr_with_target[abs(corr_with_target) >= corr_threshold].index.tolist()
        
        
        # Return subset of original DataFrame containing only features with good correlation
        self.data = self.data[good_features]
        print("The remaining columns: \n", self.data.columns.to_list() )
        print(f'Correlation threshold: {corr_threshold}')
        print(f'Size data after feature extration: {self.data.shape}')
        return self.data[good_features]
    
    def export_data(self):
        
        # Delete to ensure save only new data

        if os.path.exists(self.eda_path / "preprocessed_train.parquet"):
            os.remove(self.eda_path / "preprocessed_train.parquet")
        
        
        remove_files_in_folder(self.prob_config.train_data_path)
        

        # Export preprocessed data

        os.makedirs(self.eda_path, exist_ok=True)

        self.data.to_parquet(self.eda_path / "preprocessed_train.parquet", index=False)

        #write a .json about features input after eda
    
        raw_config = json.load(open(self.prob_config.raw_feature_config_path))
        config ={}

        config['numeric_columns'] = []
        config['category_columns']= []
        for column in self.data.columns.drop(self.target_col):
            if column in raw_config['category_columns']:
                config['category_columns'].append(column)
            else:
                config['numeric_columns'].append(column)

        config['target_column'] = raw_config['target_column'] 
        config['ml_type'] = raw_config['ml_type']


        if os.path.exists(self.eda_path / "features_config.json"):
            os.remove(self.eda_path / "features_config.json")

        with open(self.eda_path / "features_config.json", 'w+') as f:
            json.dump(config, f, indent=4)

    def prob1_process(self):

        logging.info("Extracting new feature...")
        path_save = "./src/model_config/phase-1/prob-1/sub_values.pkl"
        extractor = FeatureExtractor(self.data, path_save)
        new_data = extractor.create_new_feature(self.data)
        self.data = new_data
        logging.info("Features extracted!")
    
    def prob2_process(self):
        
        pass

    def main(self):

        self.load_data()

        self.preprocess_data(target_col = self.target_col)

        if self.prob_config.prob_id == 'prob-1':
           self.prob1_process()
           self.feature_selection()
        
        else:
           self.prob2_process()

        
        
        self.export_data()