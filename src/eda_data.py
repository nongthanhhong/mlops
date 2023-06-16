import os
import json
import yaml
import pickle
import random
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import networkx as nx
from scipy import stats
from utils import AppPath

from gensim.models import Word2Vec, KeyedVectors
from node2vec import Node2Vec
from node2vec.edges import HadamardEmbedder
from karateclub import Graph2Vec

import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils import resample
from problem_config import ProblemConfig, ProblemConst, get_prob_config, load_feature_configs_dict


class FraudEmbedding:
    def __init__(self, prob_config: ProblemConfig):

        transactions = pd.read_parquet(prob_config.captured_x_path)
        self.transactions = transactions
        self.graph = nx.DiGraph()
        
        # Create nodes for each account
        accounts = set()
        for index, transaction in self.transactions.iterrows():
            accounts.add(transaction['feature4'])
            accounts.add(transaction['feature7'])


        self.graph.add_nodes_from(accounts)
        
        # Create edges for each transaction
        for index, transaction in self.transactions.iterrows():
            self.graph.add_edge(transaction['feature4'], transaction['feature7'], amount=transaction['feature3'])
        
         # Create node embeddings
        node2vec = Node2Vec(self.graph, dimensions=64, walk_length=30, num_walks=200, workers=4)
        self.model = node2vec.fit(window=10, min_count=1)
        self.model.wv.save('./src/model_config/phase-1/prob-1/node_embeddings.bin')

    def get_embeddings(self):
        # Get embeddings for each account
        embeddings = {}
        for account in self.graph.nodes:
            embeddings[account] = self.model.wv[account]
        
        return embeddings
    
    def get_feature(self, from_account, to_account, amount):
        # Get node embeddings for from and to accounts
        
        from_embedding = self.model.wv.get_vector(from_account)
        to_embedding = self.model.wv.get_vector(to_account)
        
        # Calculate feature vector as concatenation of embeddings and transaction amount
        feature_vector = np.concatenate([from_embedding, to_embedding, [amount]])
        
        return feature_vector

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
            os.mkdir(prob_config.train_data_path)

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
            corr_threshold = np.mean(abs(corr_with_target))/10

        # Filter out features with low correlation
        good_features = corr_with_target[abs(corr_with_target) >= corr_threshold].index.tolist()
        
        
        # Return subset of original DataFrame containing only features with good correlation
        self.data = self.data[good_features]
        print("The remaining columns: ", self.data.columns )
        print(f'Correlation threshold: {corr_threshold}')
        print(f'Size data after feature extration: {self.data.shape}')
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

        # for col in categorical_cols:
        #     encoded_data[col], _ = pd.factorize(self.data[col])
        
        
        encoded_data = self.data
        
        # Fill missing values
        if fill_value == 'mean':
            encoded_data = encoded_data.fillna(encoded_data.mean())
        elif fill_value == 'median':
            encoded_data = encoded_data.fillna(encoded_data.median())
        elif isinstance(fill_value, (int, float)):
            encoded_data = encoded_data.fillna(fill_value)
        else:
            raise ValueError(f"Invalid fill value '{fill_value}'")
        
        # Remove duplicated rows
        encoded_data = encoded_data.drop_duplicates()

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
        print(len(self.data))

        if self.target_col != None: 
            data = self.data.drop([self.target_col], axis=1).copy()
        else:  data = self.data.copy()

        if method == 'z-score':

            z_scores = stats.zscore(data.select_dtypes(include=np.number))
            good_data = data[(abs(z_scores) < threshold).all(axis=1)]
            if self.target_col != None:
                
                good_data = good_data.assign(label = self.data[self.target_col])

        elif method == 'iqr':
            q1 = self.data.quantile(0.25)
            q3 = self.data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            good_data = self.data[~((self.data < lower_bound) | (self.data > upper_bound)).any(axis=1)]
        else:
            raise ValueError('Invalid method specified. Options are "z-score" and "iqr".')
        
        print(f"After handle outliers, data preserved is {len(z_scores)*100/len(self.data)}%")
        self.data = good_data
        print(len(self.data))
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

        with open(self.prob_config.train_data_path / 'types.json', 'r') as f:
           self.dtype = json.load(f)
        if drop:
            for column in self.data.columns:
                self.data[column] = pd.to_numeric(self.data[column], errors='coerce')
                self.data.dropna(inplace=True)
                self.data[column] = self.data[column].astype(self.dtype[column])
        else: 

            # doing ....
            # non tested
            for column in self.data.columns:
                self.data[column] = float(str(self.data[column]).replace(r'[^0-9.-]', ''))
        
        return self.data
    
    def export_data(self):
        
        # Delete to ensure save only new data

        if os.path.exists(self.eda_path / "preprocessed_train.parquet"):
            os.remove(self.eda_path / "preprocessed_train.parquet")
        
        
        remove_files_in_folder(self.prob_config.train_data_path)
        

        # Export preprocessed data

        os.makedirs(self.eda_path, exist_ok=True)

        self.data.to_parquet(self.eda_path / "preprocessed_train.parquet", index=False)

        #write a .json about features input after eda
    
        raw_config = json.load(open(self.prob_config.feature_config_path))
        config ={}

        config['numeric_columns'] = []
        config['category_columns']= []
        for column in self.data.columns:
            if column in raw_config['numeric_columns']:
                config['numeric_columns'].append(column)
                continue
            if column in raw_config['category_columns']:
                config['category_columns'].append(column)

        config['target_column'] = raw_config['target_column'] 
        config['ml_type'] = raw_config['ml_type']


        with open(self.eda_path / "features_config.json", 'w+') as f:
            if os.path.exists(self.eda_path / "features_config.json"):
                os.remove(self.eda_path / "features_config.json")
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
            data, _ = build_category_features(
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

    def balance_dataset(self, majority_label=0, minority_label=1, subset_percentage=0.7):
        """
        Balance an imbalanced dataset by clustering and selecting a representative subset of instances from the majority class.

        Parameters:
        -----------
        data : pandas.DataFrame
            The original dataset.
        target_col : str
            The name of the column containing the target variable.
        majority_label : int, optional (default=0)
            The label of the majority class.
        minority_label : int, optional (default=1)
            The label of the minority class.
        n_clusters : int, optional (default=3)
            The number of clusters to use when performing k-means clustering.
        subset_percentage : float, optional (default=0.5)
            The percentage of instances to select from each cluster.

        Returns:
        --------
        pandas.DataFrame
            The balanced dataset.
        """
        config_path = "./src/model_config/minibatchkmeans.yaml"
        with open(config_path, "r") as f:
            model_params = yaml.safe_load(f)

        data = self.data.copy()
        target_col = self.prob_config.target_col

        # Separate majority and minority classes
        majority_class = data[data[target_col] == majority_label]
        minority_class = data[data[target_col] == minority_label]

        # Perform clustering on the majority class instances

        k_mean = int( len(majority_class[target_col]) / 100) * len(np.unique(majority_class[target_col]))
        print(f'n_clusters = {k_mean}')
        kmeans = MiniBatchKMeans(
            n_clusters=k_mean, **model_params
            )
        kmeans.fit(majority_class.drop(target_col, axis=1))


        # Select a representative subset from each cluster
        cluster_labels = kmeans.predict(majority_class.drop(target_col, axis=1))
        majority_class_clustered = pd.concat([majority_class.reset_index(drop=True), pd.Series(cluster_labels, name='cluster')], axis=1)

        selected_instances = pd.DataFrame()
        for cluster in np.unique(cluster_labels):
            cluster_instances = majority_class_clustered[majority_class_clustered['cluster'] == cluster]
            n_instances = int(len(cluster_instances) * subset_percentage)
            selected_instances = pd.concat([selected_instances, resample(cluster_instances, n_samples=n_instances)])

        # Combine the trimmed down population with the minority class instances
        balanced_data = pd.concat([selected_instances, minority_class])
        self.data = balanced_data.drop("cluster", axis=1)
        print(len(np.unique( balanced_data["cluster"])))
        # self.data = balanced_data
        
        return balanced_data

    def add_embedding_feature(self):

        
        from_acc = self.data['feature4'].astype(float).tolist()
        to_acc = self.data['feature7'].astype(float).tolist()
        amount = self.data['feature3'].tolist()

        # print(type(from_acc[0]))
        # Load the embeddings from the file
        if os.path.isfile('./src/model_config/phase-1/prob-1/node_embeddings.bin'):
            embedding_model = KeyedVectors.load('./src/model_config/phase-1/prob-1/node_embeddings.bin')
        else: 
            FraudEmbedding(self.prob_config)
            embedding_model = KeyedVectors.load('./src/model_config/phase-1/prob-1/node_embeddings.bin')

        # feature_embedding = embedding_model.get_feature(from_acc[0], to_acc[0], amount[0])
        # Get node embeddings for from and to accounts
        
        # print(embedding_model.get_vector((str(to_acc[0]))).shape , embedding_model.get_vector((str(from_acc[0]))).shape)
        
        feature_vector = []
        for i in range(len(from_acc)):
            from_embedding = embedding_model.get_vector(str(from_acc[i]))
            to_embedding = embedding_model.get_vector((str(to_acc[i])))
            # Calculate feature vector as concatenation of embeddings and transaction amount
            feature_vector.append(np.concatenate([from_embedding, to_embedding, [amount[i]]]))

        self.data = self.data.copy().assign(node_embedding = feature_vector)
        
        return self.data 

    def input_process(self):
        self.data = self.data.drop(["batch_id", "is_drift"], axis=1)
        self.preprocess_data()
        self.handle_incorrect_format()
        self.handle_outliers()
        self.handle_incorrect_format()
        processed = eda.feature_selection()
        # balanced_data = balance_dataset(data, 'target')
        print( self.data.info())
        # self.export_data
        

        return processed

    def main(self):

        self.load_data()
        # print(self.data.info())
        
        # eda.summarize_data
        # eda.visualize_data
        
        self.preprocess_data()
        self.handle_incorrect_format()
        self.handle_outliers()

        # self.feature_selection()
        # self.balance_dataset()

        # print( self.data.info())
        if self.prob_config.prob_id == "prob-1":
            self.add_embedding_feature()

        print( self.data.info())
        
        
        self.export_data()

        # self.validate_data()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-id", type=str, default=ProblemConst.PHASE)
    parser.add_argument("--prob-id", type=str, default=ProblemConst.PROB1)
    
    args = parser.parse_args()
    
    prob_config = get_prob_config(args.phase_id, args.prob_id)

    eda = DataAnalyzer(prob_config)
    # eda.main()
    eda.load_data()
    eda.visualize_data()
    # print(eda.data.info())
    # print(eda.dtype[eda.dtype["index"] == 'feature1']["dtypes"].to_string(index=False))
