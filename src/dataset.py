import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordLevelTrainer
from torch.utils.data import Dataset, DataLoader

from tokenization import GPTSimpleTokenizer

def flatten(x):
    result = []
    for item in x:
        result.extend(item.split())
    return result

def get_sentences(ds, lang=None):
    """Yield translations or sentences in the requested language."""
    for item in ds:
        if lang is None:
            yield item["translation"]
            continue
        yield item["translation"][lang]


def truncation_and_padding(tokenizer, seq_len, pad_token="[PAD]"):
    """Configure a tokenizer to truncate and pad sequences to a fixed length."""
    # FIXME
    tokenizer.tokenizer.enable_truncation(max_length=seq_len)
    tokenizer.tokenizer.enable_padding(
        length=seq_len,
        pad_id=tokenizer.tokenizer.token_to_id(pad_token),
        pad_token=pad_token,
    )

class TranslationTokenizer:
    """Build or load a WordLevel tokenizer for a dataset."""

    def __init__(self, dataset: iter, tokenizer_path: str):
        """Initialize a tokenizer from disk or train one from ``dataset``."""
        self.ds = dataset
        # ===========setup path======
        project_root = Path(__file__).resolve().parent.parent
        artifact_dir = project_root / "dataset_artifacts" / "translation"
        self.tokenizer_path = artifact_dir / tokenizer_path

        if not self.tokenizer_path.exists():
            self.tokenizer = self.get_tokenizer()
        else:
            self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))

            self.start_token_id = self.tokenizer.token_to_id("[SOS]")
            self.end_token_id = self.tokenizer.token_to_id("[EOS]")

    def get_tokenizer(self):
        """Train, configure, save, and return a WordLevel tokenizer."""
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.normalizer = Lowercase()
        tokenizer.pre_tokenizer = Whitespace()

        trainer = WordLevelTrainer(
            special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]"],
            continuing_subword_prefix="##",
        )
        tokenizer.train_from_iterator(self.ds, trainer=trainer)

        self.start_token_id = tokenizer.token_to_id("[SOS]")
        self.end_token_id = tokenizer.token_to_id("[EOS]")

        tokenizer.post_processor = TemplateProcessing(
            single="[SOS]:0 $A:0 [EOS]:0",
            pair="[SOS]:0 $A:0 [EOS]:0 $B:1 [EOS]:1",
            special_tokens=[
                ("[SOS]", self.start_token_id),
                ("[EOS]", self.end_token_id),
            ],
        )

        tokenizer.save(str(self.tokenizer_path))

        return tokenizer

# Fix me need to tokenize on the Trainset only
class TranslationDataset(Dataset):
    
    """Prepare tokenized source and target examples for sequence-to-sequence training."""

    def __init__(self, dataset, src_lang="en", tgt_lang="nl", seq_len=100):
        """Initialize the dataset and its source and target tokenizers."""
        self.dataset = dataset

        # hard code tokenizer_name
        self.src_tokenizer = TranslationTokenizer(get_sentences(dataset, src_lang), "translation_en_tokenizer.json")
        truncation_and_padding(self.src_tokenizer, seq_len)

        self.tgt_tokenizer = TranslationTokenizer(get_sentences(dataset, tgt_lang), "translation_nl_tokenizer.json")
        # Don't intiate truncationa nd padding for decoder will be clear
        # in later partd
        # truncation_and_padding(self.tgt_tokenizer, seq_len)

        self.src_pad_id = self.src_tokenizer.tokenizer.token_to_id("[PAD]")
        self.tgt_pad_id = self.tgt_tokenizer.tokenizer.token_to_id("[PAD]")

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.src_vocab_size = self.src_tokenizer.tokenizer.get_vocab_size()
        self.tgt_vocab_size = self.tgt_tokenizer.tokenizer.get_vocab_size()

    def __len__(self):
        """Return the number of translation examples."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Return one padded source, target, label, and attention-mask example."""
        item = self.dataset[idx]

        src_text = item["translation"][self.src_lang]
        tgt_text = item["translation"][self.tgt_lang]

        src_encoded = self.src_tokenizer.tokenizer.encode(src_text).ids
        tgt_encoded = self.tgt_tokenizer.tokenizer.encode(tgt_text).ids

        # this will be already have padded sequence upto
        # sequence leng due truncation and padding
        src_ids = src_encoded[: self.seq_len]
        tgt_ids = tgt_encoded[: self.seq_len]  # upto actual length

        # =====create input and labels for decoder===========
        tgt_ids_in = tgt_ids[:-1]  # exclude <EOS>

        # this will serve output labels
        tgt_ids_out = tgt_ids[1:]  # exclude <SOS>

        # ===================================================
        # src_pad = self.seq_len - len(src_ids)
        tgt_pad = self.seq_len - len(tgt_ids_in)

        tgt_ids_in += [self.tgt_pad_id] * tgt_pad
        tgt_ids_out += [self.tgt_pad_id] * tgt_pad

        # =====================
        src_ids = torch.tensor(src_ids, dtype=torch.long)
        tgt_ids_in = torch.tensor(tgt_ids_in, dtype=torch.long)

        # get_padding mask for src
        src_mask = (src_ids != self.src_pad_id).int().unsqueeze(0).unsqueeze(0)  # (1, 1, seq_length)

        # get_padding mask for tgt
        tgt_mask = (tgt_ids_in != self.tgt_pad_id).int().unsqueeze(0).unsqueeze(0)  # (1, 1, seq_length)

        # get causal mask for tgt
        look_ahead_mask = torch.tril(torch.ones((self.seq_len, self.seq_len))).unsqueeze(
            0
        )  # (1, seq_length, seq_length)
        causal_mask = look_ahead_mask * tgt_mask  # Combine look-ahead and padding masks

        return {
            "src": src_ids,
            "tgt": tgt_ids_in,
            "label": torch.tensor(tgt_ids_out, dtype=torch.long),
            "src_mask": src_mask,
            "causal_mask": causal_mask,
        }

class GPTTextDataset(Dataset):

    def __init__(self, text: List[str], tokenizer:GPTSimpleTokenizer, seq_len:int, stride:int):
        self.in_ids = [] # input ids
        self.ta_ids = [] # tagets/labels

        text = flatten(text)
        # one thing to note here: this loop will make sure every
        # block fo texts id of length seq_len, so no 
        # post-porcesscing required to append [PAD] tokens 
        # or "<|endoftext|>" tokens. "<|endoftext|>" might be used
        # if concatinating mulitple documents, Also, in GPT
        # there is not [PAD] tokens. "<|endoftext|>" is cosnidered for
        # padding  
        for i in range(0, len(text)-seq_len, stride):
            in_seq = text[i: i+seq_len]
            ta_seq = text[i + 1: i+ seq_len + 1] # shift by 1

            self.in_ids.append(torch.tensor(tokenizer.tokenizer.encode(' '.join(in_seq)).ids, dtype= torch.long))
            self.ta_ids.append(torch.tensor(tokenizer.tokenizer.encode(' '.join(ta_seq)).ids, dtype= torch.long))

    def __len__(self):

        return len(self.in_ids)
    
    def __getitem__(self, idx):

        return self.in_ids[idx], self.ta_ids[idx]

class GPTDataModule:
    "handles  everything"

    def __init__(self,
                 seq_len: int = 512,
                 stride: int = 256,
                 batch_size: int = 16,
                 num_workers: int = 4,
            
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.ds = load_dataset("Salesforce/wikitext", name="wikitext-2-raw-v1")
        self.tokenizer = GPTSimpleTokenizer(self.ds["train"]["text"], "gpt.json")

        self.train_dataset = GPTTextDataset(self.ds["train"]["text"], self.tokenizer, seq_len, stride)
        self.val_dataset = GPTTextDataset(self.ds["val"]["text"], self.tokenizer, seq_len, stride)

    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    




if __name__ == "__main__":

    dataset = load_dataset("Salesforce/wikitext", name="wikitext-2-raw-v1", split="train")
    tokenizer = GPTSimpleTokenizer(dataset["text"], "gpt.json")
    GPTTextDataset(dataset["text"], tokenizer, seq_len=100, stride=10)
    # print(tokenizer.tokenizer.encode_batch(dataset["text"[:10]]))
    # print(dataset[2])

    # for i, t in enumerate(dataset["text"]):
    #     print(i)

    #     if i ==10:
    #         break

    #Translation dataset
    # num_rows = 38652
    # dataset = load_dataset("Helsinki-NLP/opus_books", "en-nl", split="train")
    # # a = get_sentences(dataset, "en")

    # # seems to work fine,
    # a = TranslationDataset(dataset, src_lang="en", tgt_lang="nl", seq_len=100)
    # sample = a[5]
    # assert sample["src"].shape == (100,)
    # assert sample["tgt"].shape == (100,)
    # print("TranslationDataset works:", sample["src"].shape, sample["tgt"].shape)
    # print(
    #     "Source:",
    #     a.src_tokenizer.tokenizer.decode(sample["src"].tolist(), skip_special_tokens=False),
    # )
    # print(
    #     "Target:",
    #     a.tgt_tokenizer.tokenizer.decode(sample["tgt"].tolist(), skip_special_tokens=False),
    # )
    # print(
    #     "Target Label",
    #     a.tgt_tokenizer.tokenizer.decode(sample["label"].tolist(), skip_special_tokens=False),
    # )
    # # # print(list(a))

    # # tokenizer = TranslationTokenizer(a, "translation_en-nl_tokenizer.json")
    # # tokenizer.tokenizer.enable_truncation(max_length=10)
    # # tokenizer.tokenizer.enable_padding(
    # #     length=10, pad_id=tokenizer.tokenizer.token_to_id("[PAD]"), pad_token="[PAD]"
    # # )
    # # print(tokenizer.tokenizer.encode("Let's test this tokenizer...").tokens)
    # # # batch_sentences = [
    # # #     "But what about second breakfast?",
    # # #     "Don't think he knows about second breakfast, Pip.",
    # # #     "What about elevensies?",
    # # # ]
    # # encoded_input = tokenizer.tokenizer.encode(batch_sentences)
    # # print(encoded_input.tokens)
    # print(dataset)
