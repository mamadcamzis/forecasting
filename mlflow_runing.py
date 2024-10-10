import pandas as pd

import os
from loguru import logger
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
import mlflow
import mlflow.pyfunc

from process_data import prep_store_data


class FbProphetWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
        super().__init__()
    
    def load_context(self, context):
        from bprophet import Prophet
        return

    def predict(self, context, model_input):
        future = self.model.make_future_dataframe(periods=model_input['periods'][0])
        forecast = self.model.predict(future)
        return forecast
    

if __name__ == "__main__":
    mlflow.set_experiment("Testprophet")
    file_path = 'data/train.csv'
    
    if os.path.exists(file_path):
        logger.info('Reading data from local')
        df = pd.read_csv(file_path, low_memory=False)
        prep_df = prep_store_data(df)
        print(prep_df.head())
        seasonality = {'yearly': True, 'weekly': True, 'daily': False}
        with mlflow.start_run():
            # df = pd.read_csv("data/example_wp_log_peyton_manning.csv")
            # df['y'] = df['y'].apply(lambda x: 0 if x < 0 else x)
            model = Prophet(yearly_seasonality=seasonality['yearly'],
                            weekly_seasonality=seasonality['weekly'],
                            daily_seasonality=seasonality['daily'],)
            model.fit(df)
            df_cv = cross_validation(model, initial="730 days",
                                    period="180 days",
                                        horizon="90 days")
            df_p = performance_metrics(df_cv)
            mlflow.log_metric("rmse", df_p.loc[0, 'rmse'])
            mlflow.pyfunc.log_model("model", python_model=FbProphetWrapper(model))
            print("Logged model with URI: runs:/{run_id}/model".format(run_id=mlflow.active_run().info.run_id))
        # cv_results = cross_validation(model, initial="730 days", period="180 days", horizon="365 days")
        # plot_cross_validation_metric(cv_results, metric="mape")
        # mlflow.log_artifact("fbprophet-diagnose.png")