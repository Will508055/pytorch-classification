import pandas as pd

def load_dataset(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(data):
    data['arrival_date_month_day'] = data['arrival_date_month'] + ' ' + data['arrival_date_day_of_month'].astype(str)
    data[['is_canceled', 'is_repeated_guest']] = data[['is_canceled', 'is_repeated_guest']].astype(bool)
    data['children'] = data['children'].fillna(0).astype(int)
    data[['agent', 'company']] = data[['agent', 'company']].fillna(0)
    data['country'] = data['country'].fillna('Unknown')
    data = data.drop(columns=['arrival_date_day_of_month', 'reservation_status', 'reservation_status_date'])
    return data