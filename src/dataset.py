
from datasets import load_dataset
from tokenizers import Tokenizer, PreTrainedTokenizerFast
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.normalizers import Lowercase


def get_sentences(ds, lang=None):

    for item in ds:
        if lang is None:
            yield item['translation']
        yield item['translation'][lang]
    
class BuildTokenizer:
    def __init__(self, dataset: iter, tokenizer_path: str):

        self.ds = dataset
        self.tokenizer_path = tokenizer_path

        # if tokenizer_path.exists():
        #     self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        # else:
        self.tokenizer = self.get_tokenizer()


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
                                special_tokens=[("[SOS]", self.start_token_id), ("[EOS]", self.end_token_id)])
        
        tokenizer.save(str(self.tokenizer_path))

        return tokenizer


#num_rows = 38652
dataset = load_dataset("Helsinki-NLP/opus_books", "en-nl", split="train")
a = get_sentences(dataset, "en")
# print(list(a))

tokenizer = BuildTokenizer(a, "translation_en-nl_tokenizer.json")
tokenizer.tokenizer.enable_truncation(max_length=10)
tokenizer.tokenizer.enable_padding(length=10, pad_id=tokenizer.tokenizer.token_to_id("[PAD]"), pad_token="[PAD]")
print(tokenizer.tokenizer.encode("Let's test this tokenizer...").tokens)
# batch_sentences = [
#     "But what about second breakfast?",
#     "Don't think he knows about second breakfast, Pip.",
#     "What about elevensies?",
# ]
# encoded_input = tokenizer.tokenizer.encode(batch_sentences)
# print(encoded_input.tokens)
# print(dataset)