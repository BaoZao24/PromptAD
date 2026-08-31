import torch

from models.tfam_aae import TFAMAAE, TimeFrequencyAttention, reconstruction_mse


def test_time_frequency_attention_preserves_shape():
    block = TimeFrequencyAttention(4)
    features = torch.randn(2, 4, 16, 24)
    refined, time_weight, frequency_weight = block(features)
    assert refined.shape == features.shape
    assert time_weight.shape == (2, 1, 16, 1)
    assert frequency_weight.shape == (2, 1, 1, 24)


def test_tfam_aae_reconstructs_configured_shape():
    model = TFAMAAE(
        input_shape=(32, 48),
        base_channels=4,
        latent_dim=8,
        discriminator_hidden_dim=12,
    )
    inputs = torch.rand(2, 1, 32, 48)
    reconstruction, latent = model.reconstruct(inputs)
    assert reconstruction.shape == inputs.shape
    assert latent.shape == (2, 8)
    scores = reconstruction_mse(inputs, reconstruction)
    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()
