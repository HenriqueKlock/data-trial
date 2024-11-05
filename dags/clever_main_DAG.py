from datetime import datetime, timedelta
from airflow import DAG
from airflow.utils.task_group import TaskGroup
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from scripts.clever_main_pipeline import ingest_clean_data_to_postgres, create_reviews_by_company_table, create_fmcsa_analysis_table, create_review_sentiment_table

default_args = {
    "owner": "alec.ventura",
    "start_date": datetime(2024, 10, 1),
}

datasets = [
    'fmcsa_complaints.csv',
    'fmcsa_safer_data.csv',
    'fmcsa_company_snapshot.csv',
    'fmcsa_companies.csv',
    'customer_reviews_google.csv',
    'company_profiles_google_maps.csv'
]


def run_python_operator(task_id, py_call, **kwargs) -> PythonOperator:
    return PythonOperator(
            task_id=task_id,
            python_callable=py_call,
            dag=dag,
            execution_timeout=timedelta(minutes=2),
            op_kwargs=kwargs
        )

with DAG("clever_main_DAG", default_args=default_args, catchup=False, schedule_interval='20 0 * * *', max_active_runs=1) as dag:

    start_task = EmptyOperator(task_id='Start', dag=dag)
    finish_task = EmptyOperator(task_id='Finish', dag=dag)

    with TaskGroup("ingestion") as data_ingestion_group:
        for file in datasets:
            table_name=file.split('.')[0]
            ingest_clean_data_to_postgres_task = run_python_operator(task_id=f"ingest_to_postgres_{table_name}", py_call=ingest_clean_data_to_postgres, file_name=file)

    with TaskGroup("transformations") as data_transformation_group:
        reviews_by_company_task = run_python_operator(task_id=f"reviews_by_company", py_call=create_reviews_by_company_table)
        
        reviews_by_company_task = run_python_operator(task_id=f"fmcsa_analysis", py_call=create_fmcsa_analysis_table)

    with TaskGroup("analysis") as data_analysis_group:
        reviews_by_company_task = run_python_operator(task_id=f"analyze_review_sentiment", py_call=create_review_sentiment_table)
        
    start_task >> data_ingestion_group >> data_transformation_group >> data_analysis_group >> finish_task