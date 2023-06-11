import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
from scipy import stats
import os
import pickle
import xgboost as xgb
from problem_config import ProblemConfig, ProblemConst, get_prob_config, load_feature_configs_dict
from utils import AppPath
from sklearn.model_selection import train_test_split
from raw_data_processor import RawDataProcessor
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

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



    def load_data(self):
        # Load data from path
        #save category_index.pickle file for predictor process  transform input

        training_data = pd.read_parquet(self.prob_config.raw_data_path)
        self.data, category_index = RawDataProcessor.build_category_features(
            training_data, self.prob_config.categorical_cols
        )
        self.org = training_data
        with open(self.prob_config.category_index_path, "wb") as f:
            pickle.dump(category_index, f)

        self.target_col = self.prob_config.target_col
        
    def summarize_data(self):
        """
        Summarizes the data by displaying the first few rows, the data shape, the column names, and the data types.
        
        """
        print(f'First 5 rows of data:\n{self.org.head()}')
        print(f'Data shape: {self.org.shape}')
        print(f'Column names: {self.org.columns.tolist()}')
        print(f'Data types:\n{self.org.dtypes}')
        unique_counts = {}
    
        for col in self.org.columns:
            if self.org[col].dtypes == "object":
                unique_counts[col+" - Category"] = len(self.org[col].unique())
            else: unique_counts[col] = len(self.org[col].unique())
        print(f'Count unique values:\n{unique_counts}')
        
    def visualize_data(self):
        """
        Visualizes the data by displaying histograms of all numeric columns and bar charts of all categorical columns.
        """



        data = self.org
        fraud = len(data[data[self.target_col] == 1]) / len(data) * 100
        nofraud = len(data[data[self.target_col] == 0]) / len(data) * 100
        fraud_percentage = [nofraud,fraud]

        colors = ['#FFD700','#3B3B3C']
        fig,ax = plt.subplots(nrows = 1,ncols = 2,figsize = (10,5))
        plt.subplot(1,2,1)
        plt.pie(fraud_percentage,labels = ['No Fraud', 'Fraud'],autopct='%1.1f%%',startangle = 90,colors = colors,
            wedgeprops = {'edgecolor' : 'black','linewidth': 1,'antialiased' : True})

        plt.subplot(1,2,2)
        ax = sns.countplot(x=self.target_col,data = data,edgecolor = 'black',palette = colors)
        for rect in ax.patches:
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 2, rect.get_height(), horizontalalignment='center', fontsize = 11)
        ax.set_xticklabels(['No Fraud','Fraud'])
        plt.title('Number of Fraud Cases');
        plt.show()
        
        
        # Create histograms of all numeric columns

        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        columns = 4
        if len(numeric_cols) > 4:
            row = len(numeric_cols)//4+1
        else: row = len(numeric_cols)
        plt.figure(figsize=(12,8))
        
        # set the spacing between subplots
        plt.subplots_adjust(left=0.06,
                            bottom=0,
                            right=1,
                            top=0.952,
                            wspace=0.364,
                            hspace=0.876)
        # using padding
        # fig.tight_layout(pad=5.0)   

        # plt.subplot_tool()
        for i, col in enumerate(numeric_cols):
            plt.subplot(row,columns,i+1)
            sns.histplot(data=data, x=col, kde=True)
            plt.title(f'Histogram of {col}')
        plt.show()
        
        # Create bar charts of all categorical columns
        categorical_cols = data.select_dtypes(include='object').columns.tolist()
        columns = 4
        if len(categorical_cols) > 4:
            row = len(categorical_cols)//4+1
        else: row = len(categorical_cols)
        plt.figure(figsize=(12,8))
        # set the spacing between subplots
        plt.subplots_adjust(left=0.06,
                            bottom=0,
                            right=1,
                            top=0.952,
                            wspace=0.364,
                            hspace=0.876)
        
        for i, col in enumerate(categorical_cols):
            plt.subplot(row,columns,i+1)
            sns.histplot(data=data, x=col, kde=True)
            plt.title(f'Bar Chart of {col}')
        plt.show()
  
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
            corr_threshold = np.mean(abs(corr_with_target))

        # Filter out features with low correlation
        good_features = corr_with_target[abs(corr_with_target) >= corr_threshold].index.tolist()
        
        print(corr_threshold)
        # Return subset of original DataFrame containing only features with good correlation
        self.data = self.data[good_features]
        return self.data[good_features]
    
    def preprocess_data(self, fill_value='mean', scaler=None):
        """
        Preprocesses the data by encoding categorical variables, filling missing values, and scaling the data.
        
        Args:
        categorical_cols (list): A list of the names of the categorical columns to be encoded. If None, all categorical
                                  columns will be encoded.
        fill_value (str or float): The value to use for filling missing values. If 'mean', the mean of the column will
                                    be used. If 'median', the median of the column will be used. If a float, that value
                                    will be used.
        scaler (str): The type of scaler to use for scaling the data. If None, no scaling will be performed. If 'standard',
                      the data will be standardized. If 'minmax', the data will be scaled to the range [0, 1].
        
        Returns:
        pandas.DataFrame: The preprocessed DataFrame.
        """
        
        # # Encode categorical variables
        # if categorical_cols is None:
        #     categorical_cols = self.data.select_dtypes(include='object').columns.tolist()

        encoded_data = self.data

        # for col in categorical_cols:
        #     encoded_data[col], _ = pd.factorize(self.data[col])
        
        # Fill missing values
        if fill_value == 'mean':
            encoded_data = encoded_data.fillna(encoded_data.mean())
        elif fill_value == 'median':
            encoded_data = encoded_data.fillna(encoded_data.median())
        elif isinstance(fill_value, (int, float)):
            encoded_data = encoded_data.fillna(fill_value)
        else:
            raise ValueError(f"Invalid fill value '{fill_value}'")
        
        # Scale the data
        if scaler is not None:
            if scaler == 'standard':
                encoded_data = (encoded_data - encoded_data.mean()) / encoded_data.std()
            elif scaler == 'minmax':
                encoded_data = (encoded_data - encoded_data.min()) / (encoded_data.max() - encoded_data.min())
            else:
                raise ValueError(f"Invalid scaler '{scaler}'")
        
        self.data = encoded_data
        return encoded_data
    
    def handle_outliers(self, method='z-score', threshold=3):
        """
        Detects and handles outliers in the data using the specified method.

        Args:
        method (str): The method to use for outlier detection. Options are 'z-score' and 'iqr'.
        threshold (float): The threshold used for outlier detection. For 'z-score', this is the number of standard
                            deviations from the mean. For 'iqr', this is the number of interquartile ranges from the
                            median.

        Returns:
        pandas.DataFrame: The DataFrame with the outliers removed.
        """
        data = self.data.drop([self.target_col], axis=1)
        if method == 'z-score':
            z_scores = stats.zscore(data.select_dtypes(include=np.number))
            good_data = data[(abs(z_scores) < threshold).all(axis=1)]
            good_data = pd.concat([z_scores, self.data[self.target_col]], axis=1)
        elif method == 'iqr':
            q1 = self.data.quantile(0.25)
            q3 = self.data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            good_data = self.data[~((self.data < lower_bound) | (self.data > upper_bound)).any(axis=1)]
        else:
            raise ValueError('Invalid method specified. Options are "z-score" and "iqr".')

        self.data = good_data
        return good_data

    def handle_incorrect_format(self, drop = True):

        """
        Handles incorrectly formatted data in a specified column by converting it to the correct format.
        
        Args:
        column (str): The name of the column to be corrected.
        correct_format (str): The correct format for the data in the column. Valid values are 'numeric' and 'datetime'.
        
        Returns:
        pandas.DataFrame: The DataFrame with corrected data in the specified column.
        """
        
        # if correct_format == 'numeric':
        #     # Use regex to remove non-numeric characters from column
        #     self.data[column] = self.data[column].str.replace(r'[^0-9.-]', '').astype(float)
        
        
        # elif correct_format == 'datetime':
        #     # Convert column to datetime format
        #     self.data[column] = pd.to_datetime(self.data[column])
        # else:
        #     raise ValueError(f"Invalid format '{correct_format}'")

        # Drop rows with the wrong format in each column
        if drop:
            for column in self.data.columns:
                self.data[column] = pd.to_numeric(self.data[column], errors='coerce')
            self.data.dropna(inplace=True)
        else: 

            # doing ....
            # non tested
            for column in self.data.columns:
                self.data[column] = float(str(self.data[column]).replace(r'[^0-9.-]', ''))
        
        return self.data
    
    def export_data(self):
        # Export preprocessed data
        os.makedirs(self.eda_path, exist_ok=True)

        self.data.to_parquet(self.eda_path / "preprocessed_train.parquet", index=False)

        #write a .json about features input after eda
    
        raw_config = json.load(open(prob_config.feature_config_path))
        config ={}

        config['numeric_columns'] = []
        config['category_columns']= []
        for column in self.data.columns:
            if column in raw_config['numeric_columns']:
                config['numeric_columns'].append(column)
            if column in raw_config['category_columns']:
                config['category_columns'].append(column)

        config['target_column'] = raw_config['target_column'] 
        config['ml_type'] = raw_config['ml_type']


        with open(self.eda_path / "features_config.json", 'w+') as f:
            json.dump(config, f, indent=4)

    def validate_data(self, use_eda = True):
        # Validate the data
        if self.data is None:
            print("Data is empty ???")
            return False
        if self.target_col not in self.data.columns:
            print("Can see target column in data ???")
            return False
        if self.data.isnull().values.any():
            print("Do you make sure removed all null ???")
            return False
        
        # Validate the data using XGBoost
        if use_eda:
            X = self.data.drop(columns=[self.target_col])
            y = self.data[self.target_col]
        else:
            data, _ = RawDataProcessor.build_category_features(
            self.org, self.prob_config.categorical_cols
        )
            X = data.drop(columns=[self.target_col])
            y = data[self.target_col]
        # Set XGBoost parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc'
        }

        # Train XGBoost model
        # Convert data to DMatrix format
        dmatrix = xgb.DMatrix(X, label=y)

        cv_results = xgb.cv(dtrain=dmatrix, params=params, nfold=5, num_boost_round=15, metrics='auc', as_pandas=True)
        if cv_results['test-auc-mean'].max() >= 0.99:
            print(f"Score: {cv_results['test-auc-mean'].max()} --- Good job!!!")
        else:
            print(f"Score: {cv_results['test-auc-mean'].max()} --- Harder bro!!!")

    def input_process(self):

        self.preprocess_data()
        self.handle_incorrect_format()
        processed = self.handle_outliers()

        return processed

def main(self):


    self.load_data()
    # eda.summarize_data
    # eda.visualize_data
    # print(eda.data.describe())
    
    self.preprocess_data()
    self.handle_incorrect_format()
    self.handle_outliers()
    # eda.feature_selection()

    # eda.export_data()
    self.validate_data()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-id", type=str, default=ProblemConst.PHASE)
    parser.add_argument("--prob-id", type=str, default=ProblemConst.PROB1)
    
    args = parser.parse_args()
    
    prob_config = get_prob_config(args.phase_id, args.prob_id)

    eda = DataAnalyzer(prob_config)
    eda.main()