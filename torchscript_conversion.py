import torch
from hybrid_GAN_RGB import HybridEBMGAN # Import from the RGB version which has CIFAR-10 training

# 1. Load your model with correct parameters matching the checkpoint
# The checkpoint was trained with: latent_dim=128, img_channels=3, feature_dim=128
model = HybridEBMGAN(
    latent_dim=128,
    img_channels=3,
    feature_dim=128,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
checkpoint = torch.load("cifar_EBMGAN/checkpoint_cifar_epoch_150.pth", weights_only=False)
model.generator.load_state_dict(checkpoint['generator'])
model.ebm.load_state_dict(checkpoint['ebm'])
model.generator.eval()
model.ebm.eval()

print("\nModel Architecture:")
print(f"Generator parameters: {sum(p.numel() for p in model.generator.parameters()):,}")
print(f"EBM parameters: {sum(p.numel() for p in model.ebm.parameters()):,}")

# 2. Create dummy inputs for the models
# Generator takes latent noise: [batch_size, latent_dim, 1, 1]
latent_noise = torch.randn(1, 128, 1, 1).to(model.device)

# EBM input: [batch_size, 3, 32, 32] for CIFAR-10 images
image_input = torch.randn(1, 3, 32, 32).to(model.device)

# 3. Trace the Generator
print("\nTracing Generator...")
traced_generator = torch.jit.trace(model.generator, latent_noise)
traced_generator.save("generator.pt")
print("✓ Generator saved to generator.pt")

# 4. Trace the EBM
print("Tracing EBM...")
traced_ebm = torch.jit.trace(model.ebm, image_input)
traced_ebm.save("ebm.pt")
print("✓ EBM saved to ebm.pt")

print("\nTorchScript conversion complete!")
print("Files saved:")
print("  - generator.pt (takes latent noise [1, 128, 1, 1])")
print("  - ebm.pt (takes images [1, 3, 32, 32])")

