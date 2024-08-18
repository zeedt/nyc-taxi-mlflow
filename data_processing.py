from matplotlib import pyplot as plt
import seaborn as sn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from clean_data import clean_data
import joblib


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2
 
def save_corr_map(data, name = "heat-map.png"):
    corr = data.corr()
    plt.figure(figsize=(40,20))
    plot = sn.heatmap(corr, annot=True, cmap='coolwarm')
    fig = plot.get_figure()
    fig.savefig(name)
    
def split_data(data):
    train, test = train_test_split(data,random_state=42)
    train_x = train.drop(["fare_amount"], axis=1)
    test_x = test.drop(["fare_amount"], axis=1)
    train_y = train[["fare_amount"]]
    test_y = test[["fare_amount"]]
    train_y.reset_index(drop=True)
    
    return train_x, train_y, test_x, test_y

def train_and_generate_linear_regression_model(train_x, train_y, test_x, test_y, model_name='fare_model_linear_regression.pkl'):
    lr = LinearRegression()
    lr.fit(train_x, train_y['fare_amount'].to_list())
    joblib.dump(lr, model_name)
    predicted_qualities = lr.predict(test_x)
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
    return rmse, mae, r2


def train_and_generate_decision_tree_regressor_model(train_x, train_y, test_x, test_y, model_name='fare_model_decision_tree_regressor.pkl'):
    decision_tree_model = DecisionTreeRegressor()
    decision_tree_model.fit(train_x, train_y)
    joblib.dump(decision_tree_model,model_name)
    predicted_qualities2 = decision_tree_model.predict(test_x)
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities2)
    return rmse, mae, r2

def train_and_generate_dummy_regressor_model(train_x, train_y, test_x, test_y, model_name='fare_model_dummy_regressor.pkl'):
    dummy_model = DummyRegressor()
    dummy_model.fit(train_x, train_y)
    joblib.dump(dummy_model,model_name)
    predicted_qualities2 = dummy_model.predict(test_x)
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities2)
    return rmse, mae, r2

def drop_unused_column_split_data(data):
    selected_df = data.drop(columns=['payment_type_4', 'payment_type_2','improvement_surcharge','congestion_surcharge'])
    selected_df.drop(columns=['airport_fee'], inplace=True)
        
    return split_data(selected_df)


def process_data_and_return_train_and_test_data(data):
    data = clean_data(data)
    
    save_corr_map(data)
    
    train_x, train_y, test_x, test_y = drop_unused_column_split_data(data)
    
    print(train_and_generate_linear_regression_model(train_x, train_y, test_x, test_y))
    print(train_and_generate_decision_tree_regressor_model(train_x, train_y, test_x, test_y))
    
    return train_x, train_y, test_x, test_y

