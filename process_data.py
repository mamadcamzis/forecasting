import os
import sys
import pandas as pd
import kaggle
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
from loguru import logger
from dotenv import load_dotenv
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot



load_dotenv()
KAGGLE_USERNAME = os.getenv('KAGGLE_USERNAME')
KAGGLE_KEY = os.getenv('KAGGLE_KEY')


def download_kaggle_data(kaggle_dataset: str = 'pratyushakar/rossmann-store-sales') -> None:
    api_key = kaggle.api
    logger.info('Downloading data from kaggle')
    kaggle.api.dataset_download_files(kaggle_dataset, unzip=True,
                                    path='data', quiet=False)
    logger.info('Data downloaded successfully')


def prep_store_data(df: pd.DataFrame, store_id: int = 4,
                    store_open: int = 1) -> pd.DataFrame:
    logger.info('Preprocessing store data')
    df['Date'] = pd.to_datetime(df['Date'])
    df.rename(columns={'Date': 'ds', 'Sales': 'y'}, inplace=True)
    df_store = df[(df['Store'] == store_id) & (df['Open'] == store_open)].reset_index(drop=True)
    
    return df_store.sort_values('ds', ascending=True)


def train_predict(df: pd.DataFrame,
                train_fraction: float,
                seasonality: dict) -> tuple[pd.DataFrame,
                                            pd.DataFrame,
                                            pd.DataFrame,
                                            int]:
    logger.info('Training model ...')
    train_index = int(len(df) * train_fraction)
    df_train = df.iloc[:train_index]
    df_test = df.iloc[train_index:]
    
    model = Prophet(yearly_seasonality=seasonality['yearly'],
                    weekly_seasonality=seasonality['weekly'],
                    daily_seasonality=seasonality['daily'],
                    interval_width=0.95)
    model.fit(df_train)
    predicted = model.predict(df_test)
    return predicted, df_train, df_test, train_index


def plot_forecast(df_train, df_test, predicted):
    # matplotlib.use('Agg')
    # matplotlib.use('Qt5Agg')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the training data
    ax.plot(df_train['ds'], df_train['y'], label='Train', color='blue')

    # Plot the test data
    ax.plot(df_test['ds'], df_test['y'], label='Test', color='orange')

    # Plot the predicted data
    ax.plot(predicted['ds'], predicted['yhat'], label='Predicted', color='green')

    # Add changepoints to the plot
    # add_changepoints_to_plot(fig.gca(), model, predicted)

    # Add labels and legend
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    import os

    file_path = 'data/train.csv'
    
    if os.path.exists(file_path):
        logger.info('Reading data from local')
        df = pd.read_csv(file_path, low_memory=False)
        prep_df = prep_store_data(df)
        print(prep_df.head())
        seasonality = {'yearly': True, 'weekly': True, 'daily': False}
        predicted, df_train, df_test, train_index = train_predict(prep_df, 0.8,
                                                                seasonality)
        plot_forecast(df_train, df_test, predicted)
    else:
        download_kaggle_data()
        
    
        
        

