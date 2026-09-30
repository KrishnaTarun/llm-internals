import os
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from tokenizers.processors import TemplateProcessing


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

class GPTSimpleTokenizer:
    """
    This class builds Tokenizer for training LLM model, this slighlty dfiifernt from the below 
    Tokenizer as well as what followed in GPT paper whihc use Byte-Pair-Encoding.
    """
    def __init__(self, dataset: iter, tokenizer_path: str):
            """Initialize a tokenizer from disk or train one from ``dataset``."""
            self.ds = dataset
            # ===========setup path======
            project_root = Path(__file__).resolve().parent.parent
            artifact_dir = project_root / "dataset_artifacts" / "gpt"

            if not os.path.isdir(artifact_dir):
               os.makedirs(artifact_dir, exist_ok=True)

            self.tokenizer_path = artifact_dir / tokenizer_path
            # ===========================
    
            if not self.tokenizer_path.exists():
                self.tokenizer = self.get_tokenizer()
            else:
                self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
    
    def get_tokenizer(self):
        """Train, configure, save, and return a WordLevel tokenizer."""
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.normalizer = Lowercase()
        tokenizer.pre_tokenizer = Whitespace()

        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "<|endoftext|>"],
        )
        tokenizer.train_from_iterator(self.ds, trainer=trainer)
        tokenizer.save(str(self.tokenizer_path))

        return tokenizer
