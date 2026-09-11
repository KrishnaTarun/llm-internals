
import torch
from torch import nn
from torch.utils.data import Dataset

from config import Config

class LLMTrainer:
    def __init__(self, model: nn.Module, config: Config, dataset: Dataset):

        
        self.config = config
        self.model = model.to(config.training.device)
        self.device = config.training.device 
        self.dataset = dataset




        self.criterion = nn.CrossEntropyLoss(ignore_index=self.dataset.tgt_pad_id)
        self.optimizer = torch.optim.Adam(model.parameters(), 
                                          lr=config.training.learning_rate,
                                          weight_decay=config.training.weight_decay)
        self.device = config.training.device

    def _train_epoch(self, loader):
        """
        Perform a single epoch fo training.

        Args:
            src_batch (torch.Tensor): Source batch of shape (batch_size, seq_len).
            tgt_batch (torch.Tensor): Target batch of shape (batch_size, seq_len).

        Returns:
            float: The loss value for the current training step.
        """
        self.model.train()
        total_loss = 0.0
        

        for batch in loader:
            #FIXME write assertions
            src_batch = batch["src"].to(self.device)
            tgt_batch = batch["tgt"].to(self.device)
            label = batch["label"].to(self.device)
            padding_mask = batch["src_mask"].to(self.device)
            causal_mask = batch["causal_mask"].to(self.device)

            logits = self.model(src_batch, tgt_batch, causal_mask, padding_mask)
            loss = self.criterion(logits.view(-1, self.dataset.tgt_vocab_size), label.view(-1))
            total_loss+=loss.item()

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return total_loss

    def fit(self, train_loader, val_loader):
        
        """
        Train the model for a specified number of epochs.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            num_epochs (int): Number of epochs to train the model.
        """
        #TODO: Average  meter 
        num_epochs = self.config.training.num_epochs
        train_loss = 0
        for epoch in range(num_epochs):
            train_loss = self._train_epoch(train_loader)
            avg_loss = train_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        #TODO save model
            
            
    # @torch.no_grad()
    # def evaluate(self, val_loader):