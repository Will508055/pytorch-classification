import preprocessing
import nn_architectures


if __name__ == "__main__":
    hotel_df = preprocessing.load_dataset('./hotel_bookings.csv')
    x_train, y_train, x_test, y_test = preprocessing.preprocess_data(hotel_df)