import torch
from torch.utils.data import Dataset

from utils import create_train_val_loaders


class DummyDataset(Dataset):
    """Provide small deterministic samples for data-loader tests."""

    def __init__(self):
        """Initialize twelve dummy translation examples."""
        self.items = [
            {
                "src": torch.tensor([1, 2, 3]),
                "tgt": torch.tensor([4, 5, 6]),
                "label": torch.tensor([5, 6]),
                "src_mask": torch.tensor([[[1, 1, 1]]]),
                "causal_mask": torch.tensor([[[1, 0, 0], [1, 1, 0], [1, 1, 1]]]),
            }
            for _ in range(12)
        ]

    def __len__(self):
        """Return the number of dummy examples."""
        return len(self.items)

    def __getitem__(self, idx):
        """Return the dummy example at ``idx``."""
        return self.items[idx]


def test_create_train_val_loaders_returns_expected_batches():
    """Verify loader sizes, keys, and batch dimensions."""
    dataset = DummyDataset()
    train_loader, val_loader = create_train_val_loaders(
        dataset, batch_size=4, val_ratio=0.25, seed=42
    )

    assert len(train_loader) == 9
    assert len(val_loader) == 3

    batch = next(iter(train_loader))
    assert set(batch.keys()) == {"src", "tgt", "label", "src_mask", "causal_mask"}
    assert batch["src"].shape[0] == 4
    assert batch["tgt"].shape[0] == 4
