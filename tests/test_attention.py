import torch

from model_components.attention import AttentionConfig, MultiHeadAttention


def test_attention_config_and_mha_forward_shape() -> None:
    config = AttentionConfig(model_dim=8, num_heads=2, num_kv_heads=2)
    assert config.model_dim == 8

    attention = MultiHeadAttention(model_dim=8, num_heads=2)
    x = torch.randn(2, 4, 8)
    out = attention(x, x, x)

    assert out.shape == x.shape
