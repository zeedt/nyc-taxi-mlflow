import mlflow
import pandas as pd
import glob
import os
from clean_data import clean_data
from data_processing import drop_unused_column_split_data, save_corr_map, train_and_generate_decision_tree_regressor_model, train_and_generate_linear_regression_model, train_and_generate_dummy_regressor_model
from evaluator_util import evaluate_model, predict_data_with_saved_model
from mlflow.models.signature import ModelSignature, infer_signature

raw_data = pd.concat(map(pd.read_parquet, glob.glob(os.path.join('', "/Users/saheedyusuf/Downloads/yellow-taxi-files/yellow_trip*.parquet"))))
raw_data.ffill(inplace=True)

experiment = mlflow.set_experiment(experiment_name='nyc-taxi')

base_line_model_path = 'ml-flow-dummy-regressor-mode.pkl'
base_line_artifacts = { "baseline_dummy_model" : base_line_model_path}
base_line_artifact_uri = "baseline_dummy_model_pyfunc"

data = clean_data(raw_data)
save_corr_map(data, 'lr-heat-map.png')

train_x, train_y, test_x, test_y = drop_unused_column_split_data(data)
test = test_x.copy()
test['fare_amount'] = test_y['fare_amount'].copy()

############### LINEAR REGRESSION MODEL STARTS ###############

mlflow.start_run()
mlflow.autolog(
    log_models=False,
    log_input_examples=False,
    log_model_signatures=False 
)

#### BASELINE MODEL ########
rmse, mae, r2 = train_and_generate_dummy_regressor_model(train_x, train_y, test_x, test_y, model_name='ml-flow-dummy-regressor-mode.pkl')
print(f'Dummy regressor rmse {rmse}, mae {mae}, r2 {r2}')

train_and_generate_linear_regression_model(train_x, train_y, test_x, test_y, model_name='ml-flow-trained-model-linear-regression.pkl')
mlflow.log_artifact('lr-heat-map.png')
lr_sklearn_model_path = 'ml-flow-trained-model-linear-regression.pkl'
lr_artifacts = { "lr_sklearn_model" : lr_sklearn_model_path}
lr_artifact_uri = "lr_sklearn_model_pyfunc"

predicted_qualities = predict_data_with_saved_model(lr_sklearn_model_path, test_x)
signature = infer_signature(test_x, predicted_qualities)
input_example = test_x[0:5]

evaluate_model(test, 'baseline_dummy_model', base_line_artifact_uri, base_line_artifacts, lr_artifact_uri, lr_artifacts, 'lr_sklearn_model', lr_artifact_uri, signature, input_example)

run = mlflow.last_active_run()
mlflow.register_model(
    model_uri='runs:/{}/model'.format(run.info.run_id),
    name='nyc-taxi-linear-regression-model'
)

mlflow.end_run()

############### LINEAR REGRESSION MODEL ENDS ###############

############### DECISION TREE REGRESSOR MODEL STARTS ###############
mlflow.start_run()
mlflow.autolog(
    log_models=False,
    log_input_examples=False,
    log_model_signatures=False 
)

train_and_generate_decision_tree_regressor_model(train_x, train_y, test_x, test_y, model_name='ml-flow-trained-model-decision-tree-regressor.pkl')
mlflow.log_artifact('dtr-heat-map.png')
dtr_sklearn_model_path = 'ml-flow-trained-model-decision-tree-regressor.pkl'
dtr_artifacts = { "dtr_sklearn_model" : dtr_sklearn_model_path}
dtr_artifact_uri = "dtr_sklearn_model_pyfunc"

predicted_qualities = predict_data_with_saved_model(dtr_sklearn_model_path, test_x)
signature = infer_signature(test_x, predicted_qualities)
input_example = test_x[0:5]

evaluate_model(test, 'baseline_dummy_model', base_line_artifact_uri, base_line_artifacts, dtr_artifact_uri, dtr_artifacts, 'dtr_sklearn_model', dtr_artifact_uri, signature, input_example)

run = mlflow.last_active_run()
mlflow.register_model(
    model_uri='runs:/{}/model'.format(run.info.run_id),
    name='nyc-taxi-decision-tree-regression-model'
)

mlflow.end_run()

############### DECISION TREE REGRESSOR MODEL ENDS ###############
