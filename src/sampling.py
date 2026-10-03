"""Sampling strategies for selecting tokens from model logits."""
import torch


class Sampling:
    """Select tokens using greedy, top-k, or temperature-based sampling."""

    @staticmethod
    def greedy(logits: torch.Tensor) -> torch.Tensor:
        """Return the most likely token index along the last dimension."""
        Sampling._validate_logits(logits)
        return torch.argmax(logits, dim=-1, keepdim=True)

    @staticmethod
    def probabilistic(logits: torch.Tensor) -> torch.Tensor:
        """Sample token indices from the softmax distribution over logits."""
        Sampling._validate_logits(logits)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).reshape(-1, 1)
    
    @staticmethod
    def top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
        """Sample a token after restricting logits to the top k values."""
        Sampling._validate_logits(logits)
        vocab_size = logits.shape[-1]
        if not isinstance(k, int) or not 1 <= k <= vocab_size:
            raise ValueError("k must be an integer between 1 and the vocabulary size")
        
        new_logits = torch.full_like(logits, float("-inf"))

        #gives a sorted list of the top k values and their indices along the last dimension
        values, indices = torch.topk(logits, k, dim=-1)
        new_logits[indices] = values 
        probs = torch.softmax(new_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).reshape(-1, 1)
    

    @staticmethod
    def temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Sample from the full distribution after scaling logits by temperature."""
        Sampling._validate_logits(logits)
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")

        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).reshape(-1, 1)

    @staticmethod
    def _validate_logits(logits: torch.Tensor) -> None:
        if not isinstance(logits, torch.Tensor) or logits.ndim == 0 or logits.shape[-1] == 0:
            raise ValueError("logits must be a tensor with a non-empty vocabulary dimension")


# next_token_logits = torch.tensor(
#     [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
# )

# print(Sampling.greedy(next_token_logits))  # tensor(7)
# print(Sampling.probabilistic(next_token_logits))
# print(Sampling.temperature(next_token_logits, temperature=1000))
# print(Sampling.top_k(next_token_logits, k=3))