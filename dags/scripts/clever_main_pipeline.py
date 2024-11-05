import pandas as pd
import numpy as np
from textblob import TextBlob
from scripts.postgres_helper import upload_overwrite_table, read_sql_table


def convert_to_datetime(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    for col in columns:
        try:
            df[col] = pd.to_datetime(df[col])
            print(f"Column '{col}' successfully converted to datetime.")
        except Exception as e:
            print(f"Error converting column '{col}' to datetime: {e}")
    return df


def replace_nan(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace({np.nan: None})


def create_time_for_answer_column(df: pd.DataFrame, review_date: pd.datetime, answer_date: pd.datetime) -> pd.DataFrame:
    try:
        df["days_to_answer_review"] = (df[answer_date] - df[review_date]).dt.days
    except Exception as e:
        print(f"Error creating column time_to_answer_review: {e}")
    
    return df


def group_table(df: pd.DataFrame, groupby_columns: list, aggregations: dict) -> pd.DataFrame:
    aggregation_dict = {
        new_col: (column, agg_func) for new_col, (column, agg_func) in aggregations.items()
    }
    
    return df.groupby(groupby_columns).agg(**aggregation_dict).reset_index()


def ingest_clean_data_to_postgres(**kwargs):
    file_name=kwargs.get('file_name')
    table_name=file_name.split('.')[0]
    
    raw_df = pd.read_csv(
        f'dags/scripts/data_examples/{file_name}',
        quotechar='"',
        delimiter=",",
        escapechar='\\'
    )

    df = replace_nan(raw_df)

    upload_overwrite_table(df, table_name)


def create_reviews_by_company_table():
    table_name = "trusted_reviews_by_company"

    #Read data
    company_profiles = read_sql_table("company_profiles_google_maps")
    customer_reviews = read_sql_table("customer_reviews_google")

    #Select fields
    company_profiles = company_profiles[["google_id", "name", "subtypes", "type", "category", "city", "state", "rating", "business_status", "reviews"]]
    customer_reviews = customer_reviews[["google_id", "author_title", "author_reviews_count", "review_text", "review_datetime_utc", "owner_answer", "owner_answer_timestamp_datetime_utc", "review_questions", "review_rating", "review_likes"]]

    #Adjust DateType
    customer_reviews = convert_to_datetime(customer_reviews, ["review_datetime_utc", "owner_answer_timestamp_datetime_utc"])

    #Create column days to answer review
    customer_reviews = create_time_for_answer_column(customer_reviews, "review_datetime_utc", "owner_answer_timestamp_datetime_utc")

    #Set index
    company_profiles = company_profiles.set_index("google_id")
    customer_reviews = customer_reviews.set_index("google_id")
 
    #Joining data
    df = company_profiles.join(customer_reviews, how="inner")

    upload_overwrite_table(df, table_name)
    

def create_fmcsa_analysis_table():
    table_name = "trusted_fmcsa_analysis"

    #Read data
    fmcsa_companies = read_sql_table("fmcsa_companies")
    fmcsa_company_snapshot = read_sql_table("fmcsa_company_snapshot")
    fmcsa_complaints = read_sql_table("fmcsa_complaints")
    fmcsa_safer_data = read_sql_table("fmcsa_safer_data")

    #Select fields
    fmcsa_companies = fmcsa_companies[["usdot_num", "company_name", "city", "state", "total_complaints_2021", "total_complaints_2022", "total_complaints_2023", "company_type"]]
    fmcsa_company_snapshot = fmcsa_company_snapshot[["usdot_num", "num_of_trucks", "num_of_tractors", "num_of_trailers"]]
    fmcsa_complaints = fmcsa_complaints[["usdot_num", "complaint_category", "complaint_year", "complaint_count"]]
    fmcsa_safer_data = fmcsa_safer_data[["usdot_num", "entity_type", "operating_status", "power_units", "drivers", "operation_classification", "carrier_type", "cargo_types"]]

    #Set index
    fmcsa_companies = fmcsa_companies.set_index("usdot_num")
    fmcsa_company_snapshot = fmcsa_company_snapshot.set_index("usdot_num")
    fmcsa_complaints = fmcsa_complaints.set_index("usdot_num")
    fmcsa_safer_data = fmcsa_safer_data.set_index("usdot_num")

    #Joining data
    df = fmcsa_companies.join(fmcsa_company_snapshot, how="inner").join(fmcsa_complaints, how="left").join(fmcsa_safer_data, how="left")

    upload_overwrite_table(df, table_name)


def analyze_sentiment(df: pd.DataFrame, review_column: str, sentiment_column='sentiment_score', label_column='sentiment_label') -> pd.DataFrame:

    def get_sentiment(text: str) -> str:
        try:
            polarity = TextBlob(text).sentiment.polarity
            if polarity > 0:
                label = 'positive'
            elif polarity < 0:
                label = 'negative'
            else:
                label = 'neutral'
            return polarity, label
        except Exception as e:
            print(f"Error analyzing text '{text}': {e}")
            return None, 'neutral'
    
    df[[sentiment_column, label_column]] = df[review_column].apply(lambda x: pd.Series(get_sentiment(str(x))))
    
    print(f"Sentiment analysis completed on column '{review_column}'.")
    return df


def create_review_sentiment_table():
    table_name = "refined_reviews_sentiment"

    trusted_reviews_by_company = read_sql_table("trusted_reviews_by_company")

    df = analyze_sentiment(trusted_reviews_by_company, 'review_text')

    group_columns = ["name", "category", "city", "state", "rating"]

    aggregations = {
        "avg_sentiment_score":("sentiment_score", 'mean'),
        "positive_reviews":("sentiment_label", lambda x: (x == 'positive').sum()),
        "neutral_reviews":("sentiment_label", lambda x: (x == 'neutral').sum()),
        "negative_reviews":("sentiment_label", lambda x: (x == 'negative').sum())
    }

    grouped_df = group_table(df, group_columns, aggregations)

    upload_overwrite_table(grouped_df, table_name)