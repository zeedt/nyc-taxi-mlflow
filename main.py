import mlflow
import pandas as pd
import glob
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_processing import process_data

data = pd.concat(map(pd.read_parquet, glob.glob(os.path.join('', "/Users/saheedyusuf/Downloads/yellow-taxi-files/yellow_trip*.parquet"))))
data.ffill(inplace=True)
process_data(data=data)
