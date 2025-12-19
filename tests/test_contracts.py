import torch

from src.contracts import text_encoder, unet, vae, scheduler


def test_text_encoder_contract():
    B = 1
    text_encoder.validate_inputs(torch.zeros((B, 77), dtype=torch.int64))
    text_encoder.validate_outputs(torch.zeros((B, 77, 768), dtype=torch.float32))


def test_unet_contract():
    B = 1
    unet.validate_inputs(
        torch.zeros((B, 4, 64, 64), dtype=torch.float32),
        torch.zeros((B,), dtype=torch.int64),
        torch.zeros((B, 77, 768), dtype=torch.float32),
    )
    unet.validate_outputs(torch.zeros((B, 4, 64, 64), dtype=torch.float32))


def test_vae_contract():
    B = 1
    vae.validate_inputs(torch.zeros((B, 4, 64, 64), dtype=torch.float32))
    vae.validate_outputs(torch.zeros((B, 3, 512, 512), dtype=torch.float32))


def test_scheduler_contract():
    B = 1
    scheduler.validate_step_inputs(
        torch.zeros((B, 4, 64, 64), dtype=torch.float32),
        torch.zeros((B,), dtype=torch.int64),
        torch.zeros((B, 4, 64, 64), dtype=torch.float32),
    )
    scheduler.validate_step_outputs(torch.zeros((B, 4, 64, 64), dtype=torch.float32))
