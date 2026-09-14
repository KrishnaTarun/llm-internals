import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from config import Config


class LLMTrainer:
    """Train and evaluate a sequence-to-sequence language model."""

    def __init__(self, model: nn.Module, config: Config, dataset: Dataset):
        """Initialize the model, loss function, optimizer, and training dataset."""
        self.config = config
        self.model = model.to(config.training.device)
        self.device = config.training.device
        self.dataset = dataset

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.dataset.tgt_pad_id, label_smoothing=0.1
        )
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        self.device = config.training.device

    def _train_epoch(self, loader):
        """Perform a single epoch fo training.

        Args:
            loader: DataLoader containing training batches.

        Returns:
            float: The loss value for the current training step.
        """
        self.model.train()
        total_loss = 0.0

        for batch in loader:
            # FIXME write assertions
            src_batch = batch["src"].to(self.device)
            tgt_batch = batch["tgt"].to(self.device)
            label = batch["label"].to(self.device)
            padding_mask = batch["src_mask"].to(self.device)
            causal_mask = batch["causal_mask"].to(self.device)

            logits = self.model(src_batch, tgt_batch, causal_mask, padding_mask)
            loss = self.criterion(logits.view(-1, self.dataset.tgt_vocab_size), label.view(-1))
            total_loss += loss.item()

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return total_loss / len(loader)

    def fit(self, train_loader, val_loader):
        """Train the model for a specified number of epochs.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        """
        # TODO: Average  meter
        num_epochs = self.config.training.num_epochs
        for epoch in range(num_epochs):
            train_loss = self._train_epoch(train_loader)
            self.greedy_decode(val_loader)
            # validation_loss = self.evaluate(val_loader)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}")
            # print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {validation_loss:.4f}")

        # TODO save model

    @torch.no_grad()
    def evaluate(self, val_loader):
        """Return the average loss over the validation data."""
        self.model.eval()
        val_loss = 0.0

        for batch in val_loader:
            src_batch = batch["src"].to(self.device)
            tgt_batch = batch["tgt"].to(self.device)
            label = batch["label"].to(self.device)
            padding_mask = batch["src_mask"].to(self.device)
            causal_mask = batch["causal_mask"].to(self.device)

            logits = self.model(src_batch, tgt_batch, causal_mask, padding_mask)
            val_loss += self.criterion(
                logits.view(-1, self.dataset.tgt_vocab_size), label.view(-1)
            ).item()
        return val_loss / len(val_loader)

    @torch.no_grad()
    def greedy_decode(self, loader) -> None:
        """Decode randomly selected examples from a validation or test loader.

        The loader can be replaced with a sentence-level input in a future revision.

        Args:
            loader: DataLoader containing examples to decode.
        This could be also modified for taking a sentence but its a TODO.
        """
        self.model.eval()
        # generate random sentence ids to verify
        sentence_ids = np.random.randint(0, len(loader.dataset), size=(2,))

        for sample_id in sentence_ids:
            sentence = loader.dataset[int(sample_id)]
            # Add a extra dimension for batch
            src = sentence["src"].to(self.device).unsqueeze(0)
            src_mask = sentence["src_mask"].to(self.device).unsqueeze(0)

            # get encoder out
            enc_out = self.model.encoderblock(src, src_mask)  # (b=1, seq_len, emb_dim)

            dec_in = (
                torch.empty(1, 1)
                .fill_(self.dataset.tgt_tokenizer.start_token_id)
                .type_as(src)
                .to(self.device)
            )
            while True:
                if dec_in.size(1) == self.config.data.seq_len:
                    self.idstotext(src, sentence["tgt"], dec_in.squeeze(0))
                    break
                causal_mask = torch.tril(torch.ones((dec_in.size(1), dec_in.size(1)))).unsqueeze(0)
                # unsqeeze extra to add batch diemsnion
                causal_mask = causal_mask.unsqueeze(0)

                dec_out = self.model.decoderblock(enc_out, dec_in, causal_mask, src_mask)
                prob = self.model.projection_layer(
                    dec_out[:, -1]
                )  # select the recently generated output
                _, next_word = torch.max(prob, dim=1)
                # FIXME this
                dec_in = torch.cat(
                    [
                        dec_in,
                        torch.empty(1, 1).type_as(src).fill_(next_word.item()).to(self.device),
                    ],
                    dim=1,
                )

                if next_word == self.dataset.tgt_tokenizer.end_token_id:
                    # decode and break
                    self.idstotext(
                        src, sentence["tgt"], dec_in.squeeze(0)
                    )  # get rid of batch dimension

                    break

        return

    def idstotext(self, src, tgt, gen_ids) -> None:
        """Decode and print source, target, and generated token IDs."""
        src_text = self.dataset.src_tokenizer.tokenizer.decode(
            src.squeeze(0).cpu().numpy().tolist(), skip_special_tokens=True
        )
        tgt_text = self.dataset.tgt_tokenizer.tokenizer.decode(
            tgt.cpu().numpy().tolist(), skip_special_tokens=True
        )
        gen_text = self.dataset.tgt_tokenizer.tokenizer.decode(
            gen_ids.cpu().numpy().tolist(), skip_special_tokens=True
        )
        print()
        print(f"Input Soruce text =====> {src_text}")
        print("===================================")
        print(f"Target text ===========> {tgt_text}")
        print("===================================")
        print(f"Generated text ========> {gen_text}")
        print()

        return

        # tgt_sentence = sentence["tgt"].to(self.device)
        # label = sentence["label"].to(self.device)
        # causal_mask = sentence["causal_mask"].to(self.device)
