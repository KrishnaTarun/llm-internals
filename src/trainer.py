
import numpy as np

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

        return total_loss / len(loader)

    def fit(self, train_loader, val_loader):
        
        """
        Train the model for a specified number of epochs.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            num_epochs (int): Number of epochs to train the model.
        """
        #TODO: Average  meter 
        self.greedy_decode(val_loader)
        # num_epochs = self.config.training.num_epochs
        # for epoch in range(num_epochs):

        #     train_loss = self._train_epoch(train_loader)
        #     validation_loss = self.evaluate(val_loader)
        #     print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}")
        #     print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {validation_loss:.4f}")

        #TODO save model
            
            
    @torch.no_grad()
    def evaluate(self, val_loader):
        self.model.eval()
        val_loss = 0.0

        for batch in val_loader:
            src_batch = batch["src"].to(self.device)
            tgt_batch = batch["tgt"].to(self.device)
            label = batch["label"].to(self.device)
            padding_mask = batch["src_mask"].to(self.device)
            causal_mask = batch["causal_mask"].to(self.device)

            logits = self.model(src_batch, tgt_batch, causal_mask, padding_mask)
            val_loss += self.criterion(logits.view(-1, self.dataset.tgt_vocab_size), label.view(-1)).item()
        return val_loss/ len(val_loader)

    
    @torch.no_grad()
    def greedy_decode(self,loader):

        """
        Loader here is basically a test_loader or validation loader.
        This could be also modified for taking a sentence but its a TODO.
        """

        self.model.eval()
        #generate random sentence ids to verify
        sentence_ids = np.random.randint(0, len(loader.dataset), size=(2,))
        
        for sample_id in sentence_ids:
            sentence = loader.dataset[int(sample_id)]
            #Add a extra dimension for batch
            src = sentence["src"].to(self.device).unsqueeze(0)
            src_mask = sentence["src_mask"].to(self.device).unsqueeze(0)

            #get encoder out
            enc_out = self.model.encoderblock(src, src_mask)

            dec_in = torch.empty(1,1).fill_(self.dataset.tgt_tokenizer.start_token_id).type_as(src).to(self.device)
            while True:

                if dec_in.size(1) == self.config.data.seq_len:
                   break
                causal_mask= torch.tril(torch.ones((dec_in.size(1), dec_in.size(1)))).unsqueeze(0)
                #unsqeeze extra to add batch diemsnion
                causal_mask = causal_mask.unsqueeze(0)

                dec_out = self.model.decoderblock(enc_out, dec_in, causal_mask, src_mask)
                prob = self.model.projection_layer(dec_out)
                _, next_word = torch.max(prob, dim=1)
                #FIXME this
                dec_in = torch.cat([dec_in, torch.empty(1, 1).type_as(src).fill_(next_word.item()).to(self.device)], dim=1)

                if next_word == self.dataset.tgt_tokenizer.end_token_id:
                    break

                return dec_in.squeeze(0)




            # tgt_sentence = sentence["tgt"].to(self.device)
            # label = sentence["label"].to(self.device)
            # causal_mask = sentence["causal_mask"].to(self.device) 
            


        
