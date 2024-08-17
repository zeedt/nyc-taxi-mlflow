import pandas as pd

def get_weather_period(date_str):
    value = int(str(date_str).split('-')[1])
    if value >= 3 and value < 6:
        return 'SPRING'
    elif value >= 6 and value < 9:
        return 'SUMMER'
    elif value >= 9 and value < 12:
        return 'FALL'
    else:
        return 'WINTER'
    
def get_day_period(date_str):
    value = int(str(date_str)[11:].split(':')[0])
    if value >= 0 and value < 6:
        return 'MIDNIGHT'
    elif value >= 6 and value < 12:
        return 'MORNING'
    elif value >= 12 and value < 18:
        return 'MIDDAY'
    else:
        return 'NIGHT'

def clean_data(data):
    data['day_period'] = data['tpep_pickup_datetime'].apply(get_day_period)
    data['weather_period'] = data['tpep_pickup_datetime'].apply(get_weather_period)
    data_encoded = pd.get_dummies(data, columns=['weather_period', 'day_period', 'payment_type'], dtype=int)
    data_encoded = pd.get_dummies(data_encoded, columns=['store_and_fwd_flag'], drop_first=True, dtype=int)
    cleaned_data = data_encoded[['passenger_count','trip_distance','PULocationID','DOLocationID','fare_amount','mta_tax','extra', 'tolls_amount', \
                  'improvement_surcharge', 'congestion_surcharge', 'Airport_fee', 'airport_fee', 'weather_period_FALL', \
                   'weather_period_SPRING', 'weather_period_SUMMER','weather_period_WINTER', 'day_period_MIDDAY', 'day_period_MIDNIGHT',\
       'day_period_MORNING', 'day_period_NIGHT', 'payment_type_0', 'payment_type_1', 'payment_type_2', \
                   'payment_type_3', 'payment_type_4', 'store_and_fwd_flag_Y']]
    return cleaned_data