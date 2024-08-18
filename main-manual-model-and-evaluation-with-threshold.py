import cloudpickle
import mlflow
import pandas as pd
import numpy as np
import glob
import os
import joblib
import sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from clean_data import clean_data
from data_processing import drop_unused_column_split_data, save_corr_map, train_and_generate_decision_tree_regressor_model, train_and_generate_linear_regression_model, train_and_generate_dummy_regressor_model
from mlflow.models import MetricThreshold
from mlflow.models import make_metric

raw_data = pd.concat(map(pd.read_parquet, glob.glob(os.path.join('', "/Users/saheedyusuf/Downloads/yellow-taxi-files/yellow_trip*.parquet"))))
raw_data.ffill(inplace=True)

experiment = mlflow.set_experiment(experiment_name='nyc-taxi')

conda_env = {
    "channels": ["defaults"],
    "dependencies": [
        "python={}".format(3.12),
        "pip",
        {
            "pip": [
                "mlflow=={}".format(mlflow.__version__),
                "scikit-learn=={}".format(sklearn.__version__),
                "cloudpickle=={}".format(cloudpickle.__version__),
            ],
        },
    ],
    "name": "sklearn_env",
}

class SklearnWrapper(mlflow.pyfunc.PythonModel):

    def __init__(self, artifacts_name):
        self.artifacts_name = artifacts_name

    def load_context(self, context):
        self.sklearn_model = joblib.load(context.artifacts[self.artifacts_name])

    def predict(self, context, model_input):
        return self.sklearn_model.predict(model_input.values)

################# CUSTOM METRIC START #############################################

def custom_rmse(eval_df, _builtin_metrics):
    return np.sqrt(mean_squared_error(eval_df['target'], eval_df['prediction']))

def custom_mean_absolute_error(eval_df, _builtin_metrics):
    return mean_absolute_error(eval_df['target'], eval_df['prediction'])

def custom_r2_error(eval_df, _builtin_metrics):
    return r2_score(eval_df['target'], eval_df['prediction'])

def built_in_mean_on_target_error(_eval_df, builtin_metrics):
    return builtin_metrics['mean_on_target']

################# CUSTOM METRIC END #############################################

############## Threhold definition Starts #############
thresholds = {
    "mean_absolute_error": MetricThreshold(
        threshold=9.0,  # Maximum MSE threshold
        min_absolute_change=0.1,  # Minimum absolute improvement compared to baseline
        min_relative_change=0.05,  # Minimum relative improvement compared to baseline
        greater_is_better=False  # Lower MSE is better
    )
}

############## Threhold definition Ends #############

with mlflow.start_run(experiment_id=experiment.experiment_id):
    mlflow.autolog(
        log_models=False,
        log_input_examples=False,
        log_model_signatures=False 
    )
    data = clean_data(raw_data)
    save_corr_map(data, 'lr-heat-map.png')
    
    train_x, train_y, test_x, test_y = drop_unused_column_split_data(data)
    test = test_x.copy()
    test['fare_amount'] = test_y['fare_amount'].copy()
    
    #### BASELINE MODEL ########
    rmse, mae, r2 = train_and_generate_dummy_regressor_model(train_x, train_y, test_x, test_y, model_name='ml-flow-dummy-regressor-mode.pkl')
    print(f'Dummy regressor rmse {rmse}, mae {mae}, r2 {r2}')
    
    base_line_model_path = 'ml-flow-dummy-regressor-mode.pkl'
    base_line_artifacts = { "baseline_dummy_model" : base_line_model_path}
    base_line_artifact_uri = "baseline_dummy_model_pyfunc"

    mlflow.pyfunc.log_model(
        artifact_path=base_line_artifact_uri,
        python_model=SklearnWrapper("baseline_dummy_model"),
        artifacts=base_line_artifacts,
        code_path=["main-manual-model-and-evaluation-with-threshold.py"],
        conda_env=conda_env
    )
    
    
    train_and_generate_linear_regression_model(train_x, train_y, test_x, test_y, model_name='ml-flow-trained-model-linear-regression.pkl')
    mlflow.log_artifact('lr-heat-map.png')
    dtr_sklearn_model_path = 'ml-flow-trained-model-linear-regression.pkl'
    dtr_artifacts = { "lr_sklearn_model" : dtr_sklearn_model_path}
    dtr_artifact_uri = "lr_sklearn_model_pyfunc"
    mlflow.pyfunc.log_model(
        artifact_path=dtr_artifact_uri,
        python_model=SklearnWrapper("lr_sklearn_model"),
        artifacts=dtr_artifacts,
        code_path=["main-manual-model-and-evaluation-with-threshold.py"],
        conda_env=conda_env
    )
    
    ############# Model EValuation Starts ##################################
    custom_rmse_metric = make_metric(eval_fn=custom_rmse, greater_is_better=False, name='Root Mean Square Error')
    custom_mean_absolute_error_metric = make_metric(eval_fn=custom_mean_absolute_error, greater_is_better=False, name='Mean Absolute Error')
    custom_r2_error_metric = make_metric(eval_fn=custom_r2_error, greater_is_better=False, name='Root Square Error')
    built_in_mean_on_target_error_metric = make_metric(eval_fn=built_in_mean_on_target_error, greater_is_better=True, name='Training Score')
    
    baseline_model_uri=mlflow.get_artifact_uri(base_line_artifact_uri)
    dtr_model_uri=mlflow.get_artifact_uri(dtr_artifact_uri)
    mlflow.evaluate(
        dtr_model_uri,
        test,
        targets="fare_amount",
        model_type='regressor',
        evaluators=['default'],
        custom_metrics=[
            custom_rmse_metric,
            custom_mean_absolute_error_metric,
            custom_r2_error_metric,
            built_in_mean_on_target_error_metric
        ], # Custom Metrics
        custom_artifacts=[], # Custom artifact
        validation_thresholds=thresholds,
        baseline_model=baseline_model_uri
    )
    
    ############# Model EValuation Ends ##################################

with mlflow.start_run(experiment_id=experiment.experiment_id):
    mlflow.autolog(
        log_models=False,
        log_input_examples=False,
        log_model_signatures=False 
    )
    data = clean_data(raw_data)
    save_corr_map(data, 'dtr-heat-map.png')
    
    train_x, train_y, test_x, test_y = drop_unused_column_split_data(data)
    test = test_x.copy()
    test['fare_amount'] = test_y['fare_amount'].copy()
    
    base_line_model_path = 'ml-flow-dummy-regressor-mode.pkl'
    base_line_artifacts = { "baseline_dummy_model" : base_line_model_path}
    base_line_artifact_uri = "baseline_dummy_model_pyfunc"

    mlflow.pyfunc.log_model(
        artifact_path=base_line_artifact_uri,
        python_model=SklearnWrapper("baseline_dummy_model"),
        artifacts=base_line_artifacts,
        code_path=["main-manual-model-and-evaluation-with-threshold.py"],
        conda_env=conda_env
    )
    
    
    train_and_generate_decision_tree_regressor_model(train_x, train_y, test_x, test_y, model_name='ml-flow-trained-model-decision-tree-regressor.pkl')
    mlflow.log_artifact('dtr-heat-map.png')
    dtr_sklearn_model_path = 'ml-flow-trained-model-decision-tree-regressor.pkl'
    dtr_artifacts = { "dtr_sklearn_model" : dtr_sklearn_model_path}
    dtr_artifact_uri = "dtr_sklearn_model_pyfunc"
    
    mlflow.pyfunc.log_model(
        artifact_path=dtr_artifact_uri,
        python_model=SklearnWrapper("dtr_sklearn_model"),
        artifacts=dtr_artifacts,
        code_path=["main-manual-model-and-evaluation-with-threshold.py"],
        conda_env=conda_env
    )
    
    ############# Model EValuation Starts ##################################
    custom_rmse_metric = make_metric(eval_fn=custom_rmse, greater_is_better=False, name='Root Mean Square Error')
    custom_mean_absolute_error_metric = make_metric(eval_fn=custom_mean_absolute_error, greater_is_better=False, name='Mean Absolute Error')
    custom_r2_error_metric = make_metric(eval_fn=custom_r2_error, greater_is_better=False, name='Root Square Error')
    built_in_mean_on_target_error_metric = make_metric(eval_fn=built_in_mean_on_target_error, greater_is_better=True, name='Training Score')
    
    baseline_model_uri=mlflow.get_artifact_uri(base_line_artifact_uri)
    dtr_model_uri=mlflow.get_artifact_uri(dtr_artifact_uri)
    mlflow.evaluate(
        dtr_model_uri,
        test,
        targets="fare_amount",
        model_type='regressor',
        evaluators=['default'],
        custom_metrics=[
            custom_rmse_metric,
            custom_mean_absolute_error_metric,
            custom_r2_error_metric,
            built_in_mean_on_target_error_metric
        ], # Custom Metrics
        custom_artifacts=[], # Custom artifact
        validation_thresholds=thresholds,
        baseline_model=baseline_model_uri
    )
    
    ############# Model EValuation Ends ##################################



