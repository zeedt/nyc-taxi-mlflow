import mlflow
import pandas as pd
import glob
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from clean_data import clean_data
from data_processing import drop_unused_column_split_data, process_data_and_return_train_and_test_data, save_corr_map, train_and_generate_decision_tree_regressor_model, train_and_generate_linear_regression_model

raw_data = pd.concat(map(pd.read_parquet, glob.glob(os.path.join('', "/Users/saheedyusuf/Downloads/yellow-taxi-files/yellow_trip*.parquet"))))
raw_data.ffill(inplace=True)

experiment = mlflow.get_experiment_by_name('nyc-taxi')
# if (experiment.)

with mlflow.start_run(experiment_id=experiment.experiment_id):
    mlflow.autolog()
    data = clean_data(raw_data)
    save_corr_map(data)
    
    train_x, train_y, test_x, test_y = drop_unused_column_split_data(data)
    train_and_generate_linear_regression_model(train_x, train_y, test_x, test_y, model_name='ml-flow-trained-model-linear-regression.pkl')
    # train_and_generate_decision_tree_regressor_model(train_x, train_y, test_x, test_y, model_name='ml-flow-trained-model-decision-tree-regressor.pkl')
    

with mlflow.start_run(experiment_id=experiment.experiment_id):
    mlflow.autolog()
    data = clean_data(raw_data)
    save_corr_map(data)
    
    train_x, train_y, test_x, test_y = drop_unused_column_split_data(data)
    # train_and_generate_linear_regression_model(train_x, train_y, test_x, test_y, model_name='ml-flow-trained-model-linear-regression.pkl')
    train_and_generate_decision_tree_regressor_model(train_x, train_y, test_x, test_y, model_name='ml-flow-trained-model-decision-tree-regressor.pkl')

# process_data_and_return_train_and_test_data(data=data)


