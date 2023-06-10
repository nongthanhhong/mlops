'''
                        +--------------+
                        |              |
                        |   Load Data  +
                        |              |     
                        +--------------+     
                                |
                                |
                                V
                        +--------------+           +---------------+
                        |  Preprocess  |           |               |
                        |  Data        |           |    Handle     |
                        |              |     +---->|    Outliers   |
                        |              |     |     |               |
                        |              |     |     |               |
                        +--------------+     |     +---------------+
                                |            |             |
                                |            |             |
                                V            |             V 
                        +--------------+     |     +---------------+
                        |              |     |     |               |
                        |  Handle      |     |     |   Evaluate    |
                        |  Incorrect   |-----+     |   Correlation |
                        |  Format      |           |               |
                        +--------------+           +---------------+
                                                           |
                                                           |
                        +--------------+                   |
                        |              |                   |
                        |  Export Data |<------------------+
                        |              |                   
                        +--------------+
'''


from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from problem_config import ProblemConfig, ProblemConst, get_prob_config
from eda_func import DataAnalyzer

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 6, 9),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'data_processing_dag',
    default_args=default_args,
    description='A DAG for processing data using DataAnalyzer',
    schedule_interval=timedelta(days=1)
)

# Define the DataAnalyzer object
data_analyzer = DataAnalyzer()

# Define the functions to be used in the DAG
def load_data():
    # Load data into the DataAnalyzer object
    data_analyzer.load_data('path/to/data.csv', 'target_column')

def handle_incorrect_format():
    # Handle incorrect format in the data
    data_analyzer.handle_incorrect_format('column_name')

def handle_outliers():
    # Handle outliers in the data
    data_analyzer.handle_outliers('method')

def evaluate_correlation():
    # Evaluate correlation between features
    data_analyzer.evaluate_correlation()

def preprocess_data():
    # Preprocess data
    data_analyzer.preprocess_data()

def export_data():
    # Export preprocessed data
    data_analyzer.export_data('path/to/preprocessed_data.csv')

def validate_performance():
    data_analyzer.validate_data()

# Define the tasks in the DAG
load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag
)

handle_incorrect_format_task = PythonOperator(
    task_id='handle_incorrect_format',
    python_callable=handle_incorrect_format,
    dag=dag
)

handle_outliers_task = PythonOperator(
    task_id='handle_outliers',
    python_callable=handle_outliers,
    dag=dag
)

evaluate_correlation_task = PythonOperator(
    task_id='evaluate_correlation',
    python_callable=evaluate_correlation,
    dag=dag
)

preprocess_data_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag
)

export_data_task = PythonOperator(
    task_id='export_data',
    python_callable=export_data,
    dag=dag
)

validate_performance_rask = PythonOperator(
    task_id='performance_data',
    python_callable=export_data,
    dag=dag
)


# Define the task dependencies
load_data_task >> handle_incorrect_format_task >> handle_outliers_task >> evaluate_correlation_task >> preprocess_data_task >> export_data_task