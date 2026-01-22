import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from sklearn.model_selection import StratifiedKFold
from nn_architectures import wide_nn, deep_nn


def train_model(model, x_train, y_train, x_val, y_val, print_output=False):
    loss_function = nn.BCELoss()                                                       # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.0001)                               # Adam optimizer
 
    n_epochs = 50
    batch_size = 500
    batch_start = torch.arange(0, len(x_train), batch_size)                            # Batch start indices
 
    # Hold the best model
    best_accuracy = -np.inf                                                            # Initialize to negative infinity
    best_weights = None

    # Store epoch accuracies
    epoch_accuracies = []
 
    for epoch in range(n_epochs):
        model.train()                                                                  # Set model to training mode
        
        # Shuffle the training data at the start of each epoch
        permutation = torch.randperm(x_train.size(0))
        x_train = x_train[permutation]
        y_train = y_train[permutation]
 
        for start in batch_start:
            x_batch = x_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]
            
            # Forward pass (compute predictions and loss with current weights)
            y_pred = model(x_batch)
            loss = loss_function(y_pred, y_batch)
            
            # Backward pass (compute gradients with current loss)
            optimizer.zero_grad()                                                      # Clear previous gradients
            loss.backward()
            
            # Update weights
            optimizer.step()
        
        # Evaluate on test set
        model.eval()                                                                   # Set model to evaluation mode
        with torch.no_grad():
            y_val_pred = model(x_val)
            y_val_pred_labels = (y_val_pred >= 0.5).float()
            accuracy = (y_val_pred_labels.eq(y_val).sum().item()) / len(y_val)
            epoch_accuracies.append(accuracy)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weights = copy.deepcopy(model.state_dict())                       # Save the weights of the best model

        if print_output==True:
            print(f'Epoch [{epoch+1}/{n_epochs}], Accuracy: {best_accuracy:.4f}')
    
    # Load best weights
    model.load_state_dict(best_weights)                                                # Load best model weights
    
    return model, epoch_accuracies


def kfold_cv(model, x_train, y_train):
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    cv_scores = []
    best_accuracy = 0.0

    for train_idx, test_idx in kfold.split(x_train, y_train):
        # Create model, train it, and get its accuracy for each fold
        current_model, epoch_accuracies = train_model(model, x_train[train_idx], y_train[train_idx], x_train[test_idx], y_train[test_idx])
        accuracy = epoch_accuracies[-1]
        print(f"Accuracy for fold {len(cv_scores)+1}: {accuracy:.4f}")
        cv_scores.append(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = current_model
 
    # Evaluate the model
    accuracy = np.mean(cv_scores)*100
    std = np.std(cv_scores)*100
    print(f"Model accuracy: {accuracy:.2f}% (+/- {std:.2f}%)")

    # Save the best model
    if isinstance(best_model, wide_nn):
        torch.save(best_model.state_dict(), 'wide_nn.pth')
    elif isinstance(best_model, deep_nn):
        torch.save(best_model.state_dict(), 'deep_nn.pth')
    else:
        torch.save(best_model.state_dict(), 'custom_nn.pth')

    return accuracy