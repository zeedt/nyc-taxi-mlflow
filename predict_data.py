import pandas as pd
import joblib

from clean_data import get_day_period, get_weather_period


ohe_columns = ['weather_period_FALL', 'weather_period_SPRING','weather_period_SUMMER', 'weather_period_WINTER',\
       'day_period_MIDDAY', 'day_period_MIDNIGHT', 'day_period_MORNING', 'day_period_NIGHT', 'payment_type_0', 'payment_type_1',\
       'payment_type_3', 'store_and_fwd_flag_Y']

prediction_data = pd.read_csv('nyc-test-data.csv')

# model=joblib.load(open('fare_model_linear_regression.pkl','rb'))
model=joblib.load(open('fare_model_decision_tree_regressor.pkl','rb'))

def make_prediction(prediction_data):
    if (prediction_data.columns.to_list().count('Airport_fee') < 1):
        if prediction_data.columns.to_list().count('airport_fee') < 1:
            print('Airport fee must be in file')
        else:
            prediction_data.rename(columns={'airport_fee':'Airport_fee'}, inplace=True)
            
    if (prediction_data.isna().any().to_list().count(True) > 0):
        print('None of the fields can be null. Hence not processing')
        return
    
    prediction_data['day_period'] = prediction_data['tpep_pickup_datetime'].apply(get_day_period)
    prediction_data['weather_period'] = prediction_data['tpep_pickup_datetime'].apply(get_weather_period)
    prediction_data_encoded = pd.get_dummies(prediction_data, columns=['weather_period', 'day_period', 'payment_type'], dtype=int)
    prediction_data_encoded = pd.get_dummies(prediction_data_encoded, columns=['store_and_fwd_flag'], drop_first=True, dtype=int)

    for col in ohe_columns:
        if (prediction_data_encoded.columns.to_list().count(col) == 0):
            prediction_data_encoded[col] = 0

    return model.predict(prediction_data_encoded[model.feature_names_in_])
    
    
data = make_prediction(prediction_data)

print(data)