"""
Hybrid EBM-GAN Model Implementation with CIFAR-10
==================================================
Combines Energy-Based Models with Generative Adversarial Networks
Training on CIFAR-10 dataset (32x32 RGB images).

Author: Research Experiment
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class Generator(nn.Module):
    """
    GAN Generator for CIFAR-10 (32x32 RGB images)
    Maps latent noise to realistic images
    """
    def __init__(self, latent_dim=128, img_channels=3, feature_dim=128):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        
        self.main = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, feature_dim * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_dim * 8),
            nn.ReLU(True),
            
            # State: (feature_dim*8) x 4 x 4
            nn.ConvTranspose2d(feature_dim * 8, feature_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim * 4),
            nn.ReLU(True),
            
            # State: (feature_dim*4) x 8 x 8
            nn.ConvTranspose2d(feature_dim * 4, feature_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim * 2),
            nn.ReLU(True),
            
            # State: (feature_dim*2) x 16 x 16
            nn.ConvTranspose2d(feature_dim * 2, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: img_channels x 32 x 32
        )
    
    def forward(self, z):
        return self.main(z)


class EnergyFunction(nn.Module):
    """
    Energy-Based Model for CIFAR-10 images
    Assigns scalar energy to each input (lower = more probable)
    Uses deeper architecture for better feature extraction
    """
    def __init__(self, img_channels=3, feature_dim=128):
        super(EnergyFunction, self).__init__()
        
        self.main = nn.Sequential(
            # Input: img_channels x 32 x 32
            nn.Conv2d(img_channels, feature_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            
            # State: feature_dim x 16 x 16
            nn.Conv2d(feature_dim, feature_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            
            # State: (feature_dim*2) x 8 x 8
            nn.Conv2d(feature_dim * 2, feature_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            
            # State: (feature_dim*4) x 4 x 4
            nn.Conv2d(feature_dim * 4, 1, 4, 1, 0, bias=False),
            # Output: 1 x 1 x 1 (scalar energy)
        )
    
    def forward(self, x):
        energy = self.main(x)
        return energy.view(-1)  # Return scalar energy per sample


class HybridEBMGAN:
    """
    Hybrid Model combining EBM and GAN for CIFAR-10
    
    Training Strategy:
    1. Train Generator to fool EBM (minimize energy of generated samples)
    2. Train EBM to distinguish real (low energy) from fake (high energy)
    3. Optionally refine samples using Langevin dynamics
    """
    def __init__(self, latent_dim=128, img_channels=3, feature_dim=128, 
                 device='cuda', langevin_steps=10, langevin_lr=0.01, 
                 use_spectral_norm=False):
        self.device = device
        self.latent_dim = latent_dim
        self.langevin_steps = langevin_steps
        self.langevin_lr = langevin_lr
        
        # Initialize networks
        self.generator = Generator(latent_dim, img_channels, feature_dim).to(device)
        self.ebm = EnergyFunction(img_channels, feature_dim).to(device)
        
        # Optimizers with better hyperparameters for CIFAR
        self.g_optimizer = optim.Adam(
            self.generator.parameters(), 
            lr=0.0002, 
            betas=(0.5, 0.999)
        )
        self.ebm_optimizer = optim.Adam(
            self.ebm.parameters(), 
            lr=0.0002, 
            betas=(0.5, 0.999)
        )
        
        # Initialize weights
        self.generator.apply(self._weights_init)
        self.ebm.apply(self._weights_init)
        
        print(f"\nModel Architecture:")
        print(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"EBM parameters: {sum(p.numel() for p in self.ebm.parameters()):,}")
    
    @staticmethod
    def _weights_init(m):
        """Initialize network weights"""
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    def sample_latent(self, batch_size):
        """Sample from latent space"""
        return torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)
    
    def langevin_dynamics(self, x_init, steps=None, noise_scale=1.0):
        """
        Refine samples using Langevin dynamics (MCMC sampling)
        x_new = x_old - step_size * grad_E(x) + noise
        """
        if steps is None:
            steps = self.langevin_steps
        
        x = x_init.clone().detach().requires_grad_(True)
        
        for step in range(steps):
            energy = self.ebm(x).sum()
            energy.backward()
            
            # Langevin update with decreasing noise
            noise_factor = noise_scale * (1.0 - step / steps)
            x_grad = x.grad.data
            x.data = x.data - self.langevin_lr * x_grad + \
                     torch.randn_like(x) * np.sqrt(2 * self.langevin_lr) * noise_factor
            x.grad.zero_()
            
            # Clamp to valid range
            x.data = torch.clamp(x.data, -1, 1)
        
        return x.detach()
    
    def train_step(self, real_data, use_cooperative=False):
        """
        Single training step for hybrid model
        
        Args:
            real_data: Real images from dataset
            use_cooperative: Enable cooperative training (Langevin refinement)
        
        Returns:
            Dictionary of losses and metrics
        """
        batch_size = real_data.size(0)
        
        # ============================================
        # Train EBM (Energy Function)
        # Goal: Assign low energy to real, high to fake
        # ============================================
        self.ebm_optimizer.zero_grad()
        
        # Energy of real data (should be low)
        real_energy = self.ebm(real_data)
        
        # Generate fake samples
        z = self.sample_latent(batch_size)
        fake_data = self.generator(z).detach()
        
        # Optional: Cooperative training with Langevin refinement
        if use_cooperative:
            fake_data = self.langevin_dynamics(fake_data, steps=5, noise_scale=0.5)
        
        # Energy of fake data (should be high)
        fake_energy = self.ebm(fake_data)
        
        # EBM Loss: Contrastive Divergence
        # Minimize energy of real, maximize energy of fake
        ebm_loss = real_energy.mean() - fake_energy.mean()
        
        # Add regularization to prevent energy collapse
        reg_loss = (real_energy ** 2 + fake_energy ** 2).mean()
        total_ebm_loss = ebm_loss + 0.1 * reg_loss
        
        total_ebm_loss.backward()
        self.ebm_optimizer.step()
        
        # ============================================
        # Train Generator
        # Goal: Generate samples with low energy (fool EBM)
        # ============================================
        self.g_optimizer.zero_grad()
        
        z = self.sample_latent(batch_size)
        fake_data = self.generator(z)
        fake_energy = self.ebm(fake_data)
        
        # Generator Loss: Minimize energy of generated samples
        g_loss = fake_energy.mean()
        
        g_loss.backward()
        self.g_optimizer.step()
        
        return {
            'ebm_loss': total_ebm_loss.item(),
            'g_loss': g_loss.item(),
            'real_energy': real_energy.mean().item(),
            'fake_energy': fake_energy.mean().item(),
            'energy_gap': (fake_energy.mean() - real_energy.mean()).item()
        }
    
    def generate_samples(self, num_samples, refine=False, refine_steps=None):
        """
        Generate samples from the model
        
        Args:
            num_samples: Number of samples to generate
            refine: Whether to refine with Langevin dynamics
            refine_steps: Number of Langevin steps (default: self.langevin_steps)
        """
        self.generator.eval()
        with torch.no_grad():
            z = self.sample_latent(num_samples)
            samples = self.generator(z)
        
        if refine:
            samples = self.langevin_dynamics(samples, steps=refine_steps)
        
        self.generator.train()
        return samples
    
    def save_model(self, path='hybrid_ebm_gan_cifar.pth'):
        """Save model checkpoint"""
        torch.save({
            'generator': self.generator.state_dict(),
            'ebm': self.ebm.state_dict(),
            'g_optimizer': self.g_optimizer.state_dict(),
            'ebm_optimizer': self.ebm_optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path='hybrid_ebm_gan_cifar.pth'):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator'])
        self.ebm.load_state_dict(checkpoint['ebm'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        self.ebm_optimizer.load_state_dict(checkpoint['ebm_optimizer'])
        print(f"Model loaded from {path}")


def train_hybrid_model(model, dataloader, num_epochs=100, save_interval=10, 
                       use_cooperative=False):
    """
    Training loop for hybrid EBM-GAN on CIFAR-10
    
    Args:
        model: HybridEBMGAN instance
        dataloader: PyTorch DataLoader with CIFAR-10
        num_epochs: Number of training epochs
        save_interval: Save samples every N epochs
        use_cooperative: Enable cooperative training
    """
    history = {
        'ebm_loss': [],
        'g_loss': [],
        'real_energy': [],
        'fake_energy': [],
        'energy_gap': []
    }
    
    print(f"\nTraining Configuration:")
    print(f"Cooperative Training: {use_cooperative}")
    print(f"Total Epochs: {num_epochs}")
    print(f"Save Interval: {save_interval}")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        epoch_losses = {k: [] for k in history.keys()}
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for real_data, _ in pbar:
            real_data = real_data.to(model.device)
            
            # Training step
            losses = model.train_step(real_data, use_cooperative=use_cooperative)
            
            # Accumulate losses
            for k, v in losses.items():
                epoch_losses[k].append(v)
            
            # Update progress bar
            pbar.set_postfix({
                'EBM': f"{losses['ebm_loss']:.3f}",
                'G': f"{losses['g_loss']:.3f}",
                'E_gap': f"{losses['energy_gap']:.3f}"
            })
        
        # Store epoch averages
        for k in history.keys():
            history[k].append(np.mean(epoch_losses[k]))
        
        # Generate and save samples
        if (epoch + 1) % save_interval == 0 or epoch == 0:
            print(f"\n[Epoch {epoch+1}] Generating samples...")
            
            # Standard samples
            samples = model.generate_samples(64, refine=False)
            save_samples(samples, f'cifar_samples_epoch_{epoch+1}.png', 
                        title=f'Epoch {epoch+1} - Standard')
            
            # Langevin-refined samples
            samples_refined = model.generate_samples(64, refine=True, refine_steps=20)
            save_samples(samples_refined, f'cifar_samples_refined_epoch_{epoch+1}.png',
                        title=f'Epoch {epoch+1} - Refined')
            
            # Save checkpoint
            model.save_model(f'checkpoint_cifar_epoch_{epoch+1}.pth')
        
        # Print epoch summary
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs} Summary:")
        print(f"  EBM Loss:      {history['ebm_loss'][-1]:>8.4f}")
        print(f"  Generator Loss: {history['g_loss'][-1]:>8.4f}")
        print(f"  Real Energy:   {history['real_energy'][-1]:>8.4f}")
        print(f"  Fake Energy:   {history['fake_energy'][-1]:>8.4f}")
        print(f"  Energy Gap:    {history['energy_gap'][-1]:>8.4f}")
        print(f"{'='*60}\n")
    
    return history


def save_samples(samples, filename='samples.png', nrow=8, title=None):
    """Save generated CIFAR-10 samples as image grid"""
    samples = samples.cpu()
    samples = (samples + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
    samples = samples.clamp(0, 1)
    
    fig, axes = plt.subplots(nrow, nrow, figsize=(12, 12))
    
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')
    
    for i, ax in enumerate(axes.flat):
        if i < samples.size(0):
            # Convert CHW to HWC for display
            img = samples[i].permute(1, 2, 0).numpy()
            ax.imshow(img)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def save_real_samples(dataloader, filename='cifar_real_samples.png', num_samples=64):
    """Visualize real CIFAR-10 samples"""
    # Get a batch of real images
    real_images, labels = next(iter(dataloader))
    real_images = real_images[:num_samples]
    
    # Denormalize
    real_images = (real_images + 1) / 2
    real_images = real_images.clamp(0, 1)
    
    fig, axes = plt.subplots(8, 8, figsize=(12, 12))
    fig.suptitle('Real CIFAR-10 Samples', fontsize=16, fontweight='bold')
    
    for i, ax in enumerate(axes.flat):
        if i < real_images.size(0):
            img = real_images[i].permute(1, 2, 0).numpy()
            ax.imshow(img)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved real samples to {filename}")


def plot_training_history(history, filename='cifar_training_history.png'):
    """Plot training metrics with enhanced visualization"""
    fig = plt.figure(figsize=(16, 10))
    
    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # EBM Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history['ebm_loss'], linewidth=2, color='#2E86AB')
    ax1.set_title('EBM Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    # Generator Loss
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(history['g_loss'], linewidth=2, color='#A23B72')
    ax2.set_title('Generator Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    
    # Energy Values
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(history['real_energy'], label='Real', linewidth=2, color='#06A77D')
    ax3.plot(history['fake_energy'], label='Fake', linewidth=2, color='#D5573B')
    ax3.set_title('Energy Values', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Energy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Energy Gap
    ax4 = fig.add_subplot(gs[1, :2])
    energy_gap = np.array(history['energy_gap'])
    ax4.plot(energy_gap, linewidth=2, color='#F18F01')
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax4.set_title('Energy Gap (Fake - Real)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Energy Difference')
    ax4.grid(True, alpha=0.3)
    ax4.fill_between(range(len(energy_gap)), 0, energy_gap, alpha=0.3, color='#F18F01')
    
    # Combined Loss View
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(history['ebm_loss'], label='EBM', linewidth=2, alpha=0.7)
    ax5.plot(history['g_loss'], label='Generator', linewidth=2, alpha=0.7)
    ax5.set_title('Combined Losses', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Loss')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training history to {filename}")


# ============================================
# Main Training Script
# ============================================
if __name__ == "__main__":
    # Configuration
    BATCH_SIZE = 128
    NUM_EPOCHS = 150
    LATENT_DIM = 128
    FEATURE_DIM = 128
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    USE_COOPERATIVE = False  # Set to True to enable cooperative training
    
    print("=" * 60)
    print("Hybrid EBM-GAN Training on CIFAR-10")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Number of Epochs: {NUM_EPOCHS}")
    print(f"Latent Dimension: {LATENT_DIM}")
    print(f"Feature Dimension: {FEATURE_DIM}")
    print("=" * 60)
    
    # Prepare CIFAR-10 dataset
    print("\nLoading CIFAR-10 dataset...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True if DEVICE == 'cuda' else False,
        drop_last=True
    )
    
    print(f"Dataset size: {len(train_dataset)} images")
    print(f"Number of batches: {len(dataloader)}")
    
    # CIFAR-10 classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    print(f"Classes: {', '.join(classes)}")
    
    # Save real samples for comparison
    print("\nSaving real CIFAR-10 samples...")
    save_real_samples(dataloader)
    
    # Initialize hybrid model
    print("\nInitializing Hybrid EBM-GAN model...")
    model = HybridEBMGAN(
        latent_dim=LATENT_DIM,
        img_channels=3,  # RGB
        feature_dim=FEATURE_DIM,
        device=DEVICE,
        langevin_steps=10,
        langevin_lr=0.01
    )
    
    # Train model
    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60)
    
    history = train_hybrid_model(
        model,
        dataloader,
        num_epochs=NUM_EPOCHS,
        save_interval=10,
        use_cooperative=USE_COOPERATIVE
    )
    
    # Plot results
    print("\nGenerating training history plots...")
    plot_training_history(history)
    
    # Generate final samples with different refinement levels
    print("\nGenerating final samples...")
    
    # No refinement
    final_samples = model.generate_samples(64, refine=False)
    save_samples(final_samples, 'cifar_final_samples.png', title='Final - Standard')
    
    # Light refinement
    final_samples_light = model.generate_samples(64, refine=True, refine_steps=10)
    save_samples(final_samples_light, 'cifar_final_samples_light_refined.png', 
                title='Final - Light Refined (10 steps)')
    
    # Heavy refinement
    final_samples_heavy = model.generate_samples(64, refine=True, refine_steps=30)
    save_samples(final_samples_heavy, 'cifar_final_samples_heavy_refined.png',
                title='Final - Heavy Refined (30 steps)')
    
    # Save final model
    model.save_model('final_hybrid_ebm_gan_cifar.pth')
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("Generated files:")
    print("  - cifar_real_samples.png (real CIFAR-10 images)")
    print("  - cifar_samples_epoch_*.png (samples during training)")
    print("  - cifar_samples_refined_epoch_*.png (refined samples)")
    print("  - cifar_final_samples*.png (final generated samples)")
    print("  - cifar_training_history.png (comprehensive metrics)")
    print("  - checkpoint_cifar_epoch_*.pth (model checkpoints)")
    print("  - final_hybrid_ebm_gan_cifar.pth (final model)")
    print("=" * 60)
    print("\nExperiment Tips:")
    print("1. Compare standard vs refined samples to see EBM's effect")
    print("2. Set USE_COOPERATIVE=True for cooperative training")
    print("3. Adjust langevin_steps for different refinement quality")
    print("4. Monitor energy_gap - should increase during training")
    print("=" * 60)