import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def epoch_curve(epoch_accuracies):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(epoch_accuracies)+1), epoch_accuracies)
    plt.title(f'Epoch Accuracy Curve for Final Neural Network')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.savefig('results/epoch_curve.png')
    plt.close()

def save_confusion_matrix(model, x_test, y_test):
    y_pred_np = (model(x_test) > 0.5).detach().numpy()
    y_test_np = y_test.detach().numpy()
    display = ConfusionMatrixDisplay(confusion_matrix(y_test_np, y_pred_np))
    display.plot()
    plt.title('Final Neural Network Confusion Matrix')
    plt.savefig('results/confusion_matrix.png')