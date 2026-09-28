import os
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

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
