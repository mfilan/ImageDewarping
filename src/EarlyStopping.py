import torch 

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, path, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_val_loss = None
        self.path = path
        self.stop = False
        
    def __call__(self, model, train_loss, val_loss):
        if self.best_val_loss == None:
            self.best_val_loss = val_loss
            
        elif self.best_val_loss - val_loss > self.min_delta:
            self.best_val_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
            
        elif self.best_val_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.save_checkpoint(val_loss, model)
                self.stop = True
       
    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), self.path)