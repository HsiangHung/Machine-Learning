import torch
import numpy as np

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=5, min_delta=0.0):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
                            Default: 5
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0.0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'best_model.pt'
        """
        self.patience = patience
        self.min_delta = min_delta
        
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        # First epoch
        if self.best_loss is None:
            self.best_loss = val_loss
            
        # Validation loss did NOT improve
        elif (val_loss - self.best_loss)/self.best_loss > self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                
        # Validation loss DID improve
        else:
            self.best_loss = val_loss
            self.counter = 0 # Reset counter
