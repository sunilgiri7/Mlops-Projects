from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_data
from steps.evaluate import evaluate_model
from steps.model_training import train_model

@pipeline(enable_cache=True)
def training_pipeline(data_path: str):
    df = ingest_df(data_path)
    clean_data(df)
    train_model(df)
    evaluate_model(df)