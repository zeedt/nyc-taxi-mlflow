from mlflow.models import MetricThreshold
from mlflow.models import make_metric
import joblib
import sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import cloudpickle
import mlflow

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

custom_rmse_metric = make_metric(eval_fn=custom_rmse, greater_is_better=False, name='Root Mean Square Error')
custom_mean_absolute_error_metric = make_metric(eval_fn=custom_mean_absolute_error, greater_is_better=False, name='Mean Absolute Error')
custom_r2_error_metric = make_metric(eval_fn=custom_r2_error, greater_is_better=False, name='Root Square Error')
built_in_mean_on_target_error_metric = make_metric(eval_fn=built_in_mean_on_target_error, greater_is_better=True, name='Training Score')

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

def evaluate_model(test, baseline_dummy_model_name, base_line_artifact_uri, base_line_artifacts, model_artifact_path, model_artifacts, python_model_artifact_name, model_artifact_uri, signature = None, input_example=None, code_path=[]):
    mlflow.pyfunc.log_model(
        artifact_path=base_line_artifact_uri,
        python_model=SklearnWrapper(baseline_dummy_model_name),
        artifacts=base_line_artifacts,
        code_path=code_path,
        conda_env=conda_env
    )

    mlflow.pyfunc.log_model(
        artifact_path=model_artifact_path,
        python_model=SklearnWrapper(python_model_artifact_name),
        artifacts=model_artifacts,
        code_path=code_path,
        signature = signature,
        input_example=input_example,
        conda_env=conda_env
    )

    ############# Model EValuation Starts ##################################

    baseline_model_uri=mlflow.get_artifact_uri(base_line_artifact_uri)
    model_uri = mlflow.get_artifact_uri(model_artifact_uri)
    mlflow.evaluate(
        model_uri,
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


def predict_data_with_saved_model(saved_model_path, data):
    model=joblib.load(open(saved_model_path,'rb'))
    return model.predict(data[model.feature_names_in_])
    