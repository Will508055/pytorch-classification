import preprocessing
from nn_architectures import wide_nn, deep_nn
import training
from torch import save
import plots
import metrics


if __name__ == "__main__":
    print('Starting hotel booking cancellation prediction using neural networks.\n\nPreparing data...')
    
    # Data preprocessing
    hotel_df = preprocessing.load_dataset('./hotel_bookings.csv')
    x_train, y_train, x_test, y_test = preprocessing.preprocess_data(hotel_df)

    # K-fold cross-validation to compare models
    print('\nPerforming k-fold cross-validation for wide neural network:')
    wide_nn_accuracy = training.kfold_cv(model=wide_nn(), x_train=x_train, y_train=y_train)

    print('\nPerforming k-fold cross-validation for deep neural network:')
    deep_nn_accuracy = training.kfold_cv(model=deep_nn(), x_train=x_train, y_train=y_train)

    # Train final model based on better architecture
    if wide_nn_accuracy > deep_nn_accuracy:
        print('\nWide neural network performed better. Proceeding with final model training:')
        final_model, epoch_accuracies = training.train_model(wide_nn(), x_train, y_train, x_test, y_test, print_output=True)
        save(final_model.state_dict(), 'results/final_nn.pth')
    else:
        print('\nDeep neural network performed better. Proceeding with final model training:')
        final_model, epoch_accuracies = training.train_model(deep_nn(), x_train, y_train, x_test, y_test, print_output=True)
        save(final_model.state_dict(), 'results/final_nn.pth')

    # Save diagnostic plots
    plots.epoch_curve(epoch_accuracies)
    plots.save_confusion_matrix(final_model, x_test, y_test)

    # Save evaluation metrics
    metrics.calculate_metrics(final_model, x_test, y_test)
    print('\nFinal model, diagnostic plots, and evaluation metrics have been saved in the results/ directory.')



    


