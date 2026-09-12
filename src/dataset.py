from  pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.normalizers import Lowercase

import torch
from torch.utils.data import Dataset


def get_sentences(ds, lang=None):

    for item in ds:
        if lang is None:
            yield item['translation']
        yield item['translation'][lang]

def truncation_and_padding(tokenizer, seq_len, pad_token="[PAD]"):
    #FIXME
    tokenizer.tokenizer.enable_truncation(max_length=seq_len)
    tokenizer.tokenizer.enable_padding(length=seq_len, pad_id=tokenizer.tokenizer.token_to_id(pad_token), pad_token=pad_token)

class BuildTokenizer:
    def __init__(self, dataset: iter, tokenizer_path: str):

        self.ds = dataset
        self.tokenizer_path = Path(__file__).resolve().parent.parent / tokenizer_path

        if not self.tokenizer_path.exists():
            self.tokenizer = self.get_tokenizer()
        else:
            self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))

            self.start_token_id = self.tokenizer.token_to_id("[SOS]")
            self.end_token_id = self.tokenizer.token_to_id("[EOS]")


    def get_tokenizer(self):

        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.normalizer =Lowercase()
        tokenizer.pre_tokenizer = Whitespace()

        trainer = WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]"],
                                   continuing_subword_prefix="##")
        tokenizer.train_from_iterator(self.ds, trainer=trainer)

        self.start_token_id = tokenizer.token_to_id("[SOS]")
        self.end_token_id = tokenizer.token_to_id("[EOS]")

        tokenizer.post_processor = TemplateProcessing(
                                single=f"[SOS]:0 $A:0 [EOS]:0",
                                pair=f"[SOS]:0 $A:0 [EOS]:0 $B:1 [EOS]:1",
                                special_tokens=[("[SOS]", self.start_token_id), ("[EOS]", self.end_token_id)])
        
        tokenizer.save(str(self.tokenizer_path))

        return tokenizer

class TranslationDataset(Dataset):
    def __init__(self, dataset, src_lang="en", tgt_lang="nl", seq_len=100):

        self.dataset = dataset

        #hard code tokenizer_name
        self.src_tokenizer = BuildTokenizer(get_sentences(dataset, src_lang), 
                                            "translation_en_tokenizer.json")
        truncation_and_padding(self.src_tokenizer, seq_len)

        self.tgt_tokenizer = BuildTokenizer(get_sentences(dataset, tgt_lang),
                                                                    "translation_nl_tokenizer.json")
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
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        src_text = item['translation'][self.src_lang]
        tgt_text = item['translation'][self.tgt_lang]

        src_encoded = self.src_tokenizer.tokenizer.encode(src_text).ids
        tgt_encoded = self.tgt_tokenizer.tokenizer.encode(tgt_text).ids

        # this will be already have padded sequence upto 
        # sequence leng due truncation and padding
        src_ids = src_encoded[:self.seq_len]
        tgt_ids = tgt_encoded[:self.seq_len] #upto actual length 

        #=====create input and labels for decoder===========
        tgt_ids_in =  tgt_ids[:-1] # exclude <EOS>

        #this will serve output labels
        tgt_ids_out = tgt_ids[1:]  # exclude <SOS>

        #===================================================
        # src_pad = self.seq_len - len(src_ids)
        tgt_pad = self.seq_len - len(tgt_ids_in)

        
        tgt_ids_in  += [self.tgt_pad_id] * tgt_pad
        tgt_ids_out += [self.tgt_pad_id] * tgt_pad

        #=====================
        src_ids = torch.tensor(src_ids, dtype=torch.long)
        tgt_ids_in = torch.tensor(tgt_ids_in, dtype=torch.long)

        # get_padding mask for src
        src_mask = (src_ids!= self.src_pad_id).int().unsqueeze(0).unsqueeze(0)  # (1, 1, seq_length)

        # get_padding mask for tgt
        tgt_mask = (tgt_ids_in!= self.tgt_pad_id).int().unsqueeze(0).unsqueeze(0)  # (1, 1, seq_length)
        
        # get causal mask for tgt
        look_ahead_mask = torch.tril(torch.ones((self.seq_len, self.seq_len))).unsqueeze(0) # (1, seq_length, seq_length)
        causal_mask = look_ahead_mask * tgt_mask  # Combine look-ahead and padding masks

        return {
            "src": src_ids,
            "tgt": tgt_ids_in,
            "label": torch.tensor(tgt_ids_out, dtype=torch.long),
            "src_mask": src_mask,
            "causal_mask": causal_mask
        }

if __name__ == "__main__":
    #num_rows = 38652
    dataset = load_dataset("Helsinki-NLP/opus_books", "en-nl", split="train")
    # a = get_sentences(dataset, "en")

    #seems to work fine,
    a = TranslationDataset(dataset, src_lang="en", tgt_lang="nl", seq_len=100)
    sample = a[5]
    assert sample["src"].shape == (100,)
    assert sample["tgt"].shape == (100,)
    print("TranslationDataset works:", sample["src"].shape, sample["tgt"].shape)
    print("Source:", a.src_tokenizer.tokenizer.decode(sample["src"].tolist(), skip_special_tokens=False))
    print("Target:", a.tgt_tokenizer.tokenizer.decode(sample["tgt"].tolist(), skip_special_tokens=False))
    print("Target Label", a.tgt_tokenizer.tokenizer.decode(sample["label"].tolist(), skip_special_tokens=False))
    # # print(list(a))

    # tokenizer = BuildTokenizer(a, "translation_en-nl_tokenizer.json")
    # tokenizer.tokenizer.enable_truncation(max_length=10)
    # tokenizer.tokenizer.enable_padding(length=10, pad_id=tokenizer.tokenizer.token_to_id("[PAD]"), pad_token="[PAD]")
    # print(tokenizer.tokenizer.encode("Let's test this tokenizer...").tokens)
    # # batch_sentences = [
    # #     "But what about second breakfast?",
    # #     "Don't think he knows about second breakfast, Pip.",
    # #     "What about elevensies?",
    # # ]
    # encoded_input = tokenizer.tokenizer.encode(batch_sentences)
    # print(encoded_input.tokens)
    # print(dataset)