from category_encoders import OneHotEncoder, TargetEncoder
import pandas as pd
import numpy as np
from category_encoders import TargetEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from transformers import pipeline
import torch


def load_dataset(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(data):
    # Make date field
    data['arrival_date_month_day'] = data['arrival_date_month'] + ' ' + data['arrival_date_day_of_month'].astype(str)

    # Change data types and handle missing values
    data[['is_canceled', 'is_repeated_guest']] = data[['is_canceled', 'is_repeated_guest']].astype(bool)
    data['children'] = data['children'].fillna(0).astype(int)
    data[['agent', 'company']] = data[['agent', 'company']].fillna(0)
    data['country'] = data['country'].fillna('Unknown')

    # Remove unnecessary columns
    data = data.drop(columns=['arrival_date_day_of_month', 'reservation_status', 'reservation_status_date'])
    
    # Get target variable
    y = data['is_canceled']
    data = data.drop(columns=['is_canceled'])

    # Identify column types for transformations
    numerical_cols = data.select_dtypes(include=np.number).columns.tolist()
    onehot_cols = ['hotel', 'arrival_date_month', 'meal', 'market_segment', 'distribution_channel', 'reserved_room_type',
                   'assigned_room_type', 'deposit_type', 'customer_type']
    target_encode_cols = ['country', 'agent', 'company', 'arrival_date_month_day']

    # Create ColumnTransformer with one-hot encoding, target encoding, and scaling
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(), onehot_cols),
            ('target', TargetEncoder(), target_encode_cols),
            ('scaler', StandardScaler(), numerical_cols)
        ],
        remainder='passthrough'
    )

    # Create pipeline with imputation and scaling (shouldn't be strictly necessary after preprocessor, but just to be sure)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Split data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=123)

    # Fit pipeline on training data before applying it to both data sets
    pipeline.fit(x_train, y_train)
    x_train = pipeline.transform(x_train)
    x_test = pipeline.transform(x_test)

    # Convert all data to PyTorch tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)

    return x_train, y_train, x_test, y_test