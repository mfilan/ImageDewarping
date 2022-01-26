import numpy as np
import torch
import wandb
from EarlyStopping import EarlyStopping

class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 training_DataLoader: torch.utils.data.Dataset,
                 validation_DataLoader: torch.utils.data.Dataset = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 epochs: int = 100,
                 epoch: int = 0,
                 notebook: bool = False,
                 use_wandb: bool = False,
                 use_early_stopping: bool = True,
                 project_name: str = "image dewarping",
                 save_model_name: str = "model.pth"
                 ):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.device = device
        self.epochs = epochs
        self.epoch = epoch
        self.notebook = notebook
        self.use_wandb = use_wandb
        self.use_early_stopping = use_early_stopping
        self.save_model_name = save_model_name
        
        
        if self.use_wandb:
            wandb.init(project=project_name)
            wandb.define_metric("validation_loss", step_metric="step")
         
        if self.use_early_stopping:
            self.early_stopping = EarlyStopping(path = self.save_model_name)
        
        self.training_step = 0
        self.training_loss = []
        self.validation_loss = []
        self.learning_rate = []

    def train_log(self,loss):
    # Where the magic happens
        wandb.log({"epoch": self.epoch, "loss": loss}, step=self.training_step)
        
    def validation_log(self,loss):
    # Where the magic happens
        wandb.log({"epoch": self.epoch,"validation_loss": loss}, step=self.training_step)
        
    def run_trainer(self):
        try:
            return self._run_trainer()
        except KeyboardInterrupt:
            print("KeyboardInterrupt detected saving model...!")
            torch.save(self.model.state_dict(), self.save_model_name)
            if self.use_wandb:
                wandb.finish()
        
    def _run_trainer(self):
        
        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange
        
        if self.use_wandb:
            wandb.watch(self.model, self.criterion, log="all", log_freq=10)

        progressbar = trange(self.epochs, desc='Progress')
        for i in progressbar:
            """Epoch counter"""
            self.epoch += 1  # epoch counter

            """Training block"""
            self._train()

            """Validation block"""
            if self.validation_DataLoader is not None:
                self._validate()
            """Early stopping"""
            if self.use_early_stopping:
                self.early_stopping(self.model, self.training_loss[-1], self.validation_loss[-1])
                if self.early_stopping.stop:
                    break
            """Learning rate scheduler block"""
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()  # learning rate scheduler step
        wandb.finish()
        return self.training_loss, self.validation_loss, self.learning_rate

    def _train(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.train()  # train mode
        train_losses = []  # accumulate the losses here
        batch_iter = tqdm(enumerate(self.training_DataLoader), 'Training', total=len(self.training_DataLoader),
                          leave=False)

        for i, (x, y) in batch_iter:
            input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)
            self.optimizer.zero_grad()  # zerograd the parameters
            if self.model.__class__.__name__ == "DewarpingUNet":
                out1,out2 = self.model(input)  # one forward pass
                loss = self.criterion(out1,out2, target)  # calculate loss 
            else:
                out = self.model(input)  # one forward pass
                loss = self.criterion(out, target)  # calculate loss
            loss_value = loss.item()
            train_losses.append(loss_value)
            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters
            if self.use_wandb:
                self.train_log(loss_value)
            batch_iter.set_description(f'Training: (loss {loss_value:.4f})')  # update progressbar
            self.training_step +=1
        self.training_loss.append(np.mean(train_losses))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

        batch_iter.close()

    def _validate(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses here
        batch_iter = tqdm(enumerate(self.validation_DataLoader), 'Validation', total=len(self.validation_DataLoader),
                          leave=False)

        for i, (x, y) in batch_iter:
            input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)

            with torch.no_grad():
                if self.model.__class__.__name__ == "DewarpingUNet":
                    out1,out2 = self.model(input)  # one forward pass
                    loss = self.criterion(out1,out2, target)  # calculate loss 
                else:
                    out = self.model(input)
                    loss = self.criterion(out, target)
                loss_value = loss.item()
                valid_losses.append(loss_value)
                batch_iter.set_description(f'Validation: (loss {loss_value:.4f})')
        mean_losses = np.mean(valid_losses)
        if self.use_wandb:
            self.validation_log(mean_losses)

        self.validation_loss.append(mean_losses)

        batch_iter.close()