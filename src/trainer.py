
import torch
from torch import nn

from config import Config

class Trainer:
    def __init__(self, model: nn.Module, config: Config):
        self.model = model
        self.criterion = nn.CrossEntropyLoss(ignore_index=config.data.tgt_pad_idx)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
        self.device = config.training.device

    def _train_step(self, src_batch, tgt_batch):
        """
        Perform a single training step.

        Args:
            src_batch (torch.Tensor): Source batch of shape (batch_size, seq_len).
            tgt_batch (torch.Tensor): Target batch of shape (batch_size, seq_len).

        Returns:
            float: The loss value for the current training step.
        """
        self.model.train()
        src_batch = src_batch.to(self.device)
        tgt_batch = tgt_batch.to(self.device)

        # Forward pass
        output = self.model(src_batch, tgt_batch[:, :-1])  # Exclude the last token for teacher forcing
        loss = self.criterion(output.reshape(-1, output.size(-1)), tgt_batch[:, 1:].reshape(-1))  # Shift target for loss calculation

        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def fit(self, train_loader, num_epochs):
        """
        Train the model for a specified number of epochs.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            num_epochs (int): Number of epochs to train the model.
        """
        for epoch in range(num_epochs):
            total_loss = 0
            for src_batch, tgt_batch in train_loader:
                loss = self._train_step(src_batch, tgt_batch)
                total_loss += loss

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
            
    @torch.no_grad()
    def evaluate(self, val_loader):