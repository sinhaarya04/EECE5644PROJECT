"""
Convolutional Conditional Neural Process (ConvCNP) implementation for image inpainting
Based on: "Convolutional Conditional Neural Processes" by Gordon et al. (2019)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvCNP(nn.Module):
    """
    Convolutional Conditional Neural Process for image inpainting.
    
    Uses convolutional layers to maintain translation equivariance,
    making it more suitable for image data than standard CNP.
    """
    
    def __init__(
        self,
        in_channels=3,  # RGB input
        hidden_channels=64,
        num_layers=4,
        kernel_size=3,
        padding_mode='reflect',
        use_residual=True
    ):
        super(ConvCNP, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.use_residual = use_residual
        
        # Encoder: process context image (masked image)
        encoder_layers = []
        
        # First layer: input -> hidden
        encoder_layers.append(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, 
                     padding=kernel_size//2, padding_mode=padding_mode)
        )
        encoder_layers.append(nn.BatchNorm2d(hidden_channels))
        encoder_layers.append(nn.ReLU(inplace=True))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            encoder_layers.append(
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size,
                         padding=kernel_size//2, padding_mode=padding_mode)
            )
            encoder_layers.append(nn.BatchNorm2d(hidden_channels))
            encoder_layers.append(nn.ReLU(inplace=True))
        
        # Last encoder layer
        encoder_layers.append(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size,
                     padding=kernel_size//2, padding_mode=padding_mode)
        )
        encoder_layers.append(nn.BatchNorm2d(hidden_channels))
        encoder_layers.append(nn.ReLU(inplace=True))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder: predict RGB values
        decoder_layers = []
        
        # First decoder layer
        decoder_layers.append(
            nn.Conv2d(hidden_channels + 1, hidden_channels, kernel_size=kernel_size,
                     padding=kernel_size//2, padding_mode=padding_mode)
        )
        decoder_layers.append(nn.BatchNorm2d(hidden_channels))
        decoder_layers.append(nn.ReLU(inplace=True))
        
        # Hidden decoder layers
        for _ in range(num_layers - 2):
            decoder_layers.append(
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size,
                         padding=kernel_size//2, padding_mode=padding_mode)
            )
            decoder_layers.append(nn.BatchNorm2d(hidden_channels))
            decoder_layers.append(nn.ReLU(inplace=True))
        
        # Output layer: predict RGB
        decoder_layers.append(
            nn.Conv2d(hidden_channels, in_channels, kernel_size=kernel_size,
                     padding=kernel_size//2, padding_mode=padding_mode)
        )
        decoder_layers.append(nn.Sigmoid())  # Ensure output in [0, 1]
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, masked_image, mask):
        """
        Forward pass through ConvCNP.
        
        Args:
            masked_image: Masked input image [B, C, H, W] in range [0, 1]
            mask: Binary mask [B, 1, H, W] where 1 = masked (to predict), 0 = context
        
        Returns:
            pred_image: Predicted full image [B, C, H, W]
            pred_masked: Predicted values only in masked regions [B, C, H, W]
        """
        # Encode the masked image
        encoded = self.encoder(masked_image)  # [B, hidden_channels, H, W]
        
        # Concatenate mask as additional channel for decoder
        # This helps the model know which regions to predict
        decoder_input = torch.cat([encoded, mask], dim=1)  # [B, hidden_channels+1, H, W]
        
        # Decode to get full image prediction
        pred_image = self.decoder(decoder_input)  # [B, C, H, W]
        
        # For masked regions, use prediction; for context, use original
        # mask: 1 = masked (use prediction), 0 = context (use original)
        pred_masked = pred_image * mask + masked_image * (1 - mask)
        
        return pred_image, pred_masked


class ConvCNPWithUncertainty(nn.Module):
    """
    ConvCNP variant that also predicts uncertainty.
    """
    
    def __init__(
        self,
        in_channels=3,
        hidden_channels=64,
        num_layers=4,
        kernel_size=3,
        padding_mode='reflect',
        use_residual=True
    ):
        super(ConvCNPWithUncertainty, self).__init__()
        
        # Encoder (same as ConvCNP)
        encoder_layers = []
        encoder_layers.append(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size,
                     padding=kernel_size//2, padding_mode=padding_mode)
        )
        encoder_layers.append(nn.BatchNorm2d(hidden_channels))
        encoder_layers.append(nn.ReLU(inplace=True))
        
        for _ in range(num_layers - 2):
            encoder_layers.append(
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size,
                         padding=kernel_size//2, padding_mode=padding_mode)
            )
            encoder_layers.append(nn.BatchNorm2d(hidden_channels))
            encoder_layers.append(nn.ReLU(inplace=True))
        
        encoder_layers.append(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size,
                     padding=kernel_size//2, padding_mode=padding_mode)
        )
        encoder_layers.append(nn.BatchNorm2d(hidden_channels))
        encoder_layers.append(nn.ReLU(inplace=True))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Shared decoder features
        shared_layers = []
        shared_layers.append(
            nn.Conv2d(hidden_channels + 1, hidden_channels, kernel_size=kernel_size,
                     padding=kernel_size//2, padding_mode=padding_mode)
        )
        shared_layers.append(nn.BatchNorm2d(hidden_channels))
        shared_layers.append(nn.ReLU(inplace=True))
        
        for _ in range(num_layers - 2):
            shared_layers.append(
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size,
                         padding=kernel_size//2, padding_mode=padding_mode)
            )
            shared_layers.append(nn.BatchNorm2d(hidden_channels))
            shared_layers.append(nn.ReLU(inplace=True))
        
        self.shared_decoder = nn.Sequential(*shared_layers)
        
        # Mean head
        self.mean_head = nn.Sequential(
            nn.Conv2d(hidden_channels, in_channels, kernel_size=kernel_size,
                     padding=kernel_size//2, padding_mode=padding_mode),
            nn.Sigmoid()
        )
        
        # Variance head
        self.var_head = nn.Sequential(
            nn.Conv2d(hidden_channels, in_channels, kernel_size=kernel_size,
                     padding=kernel_size//2, padding_mode=padding_mode),
            nn.Softplus()
        )
    
    def forward(self, masked_image, mask):
        """
        Forward pass with uncertainty prediction.
        
        Returns:
            pred_image: Predicted image [B, C, H, W]
            pred_masked: Predicted values in masked regions [B, C, H, W]
            sigma: Uncertainty map [B, C, H, W]
        """
        # Encode
        encoded = self.encoder(masked_image)
        
        # Decode shared features
        decoder_input = torch.cat([encoded, mask], dim=1)
        shared_features = self.shared_decoder(decoder_input)
        
        # Predict mean and variance
        mu = self.mean_head(shared_features)
        log_var = self.var_head(shared_features)
        sigma = torch.sqrt(log_var + 1e-6)
        
        # Combine with original for masked regions
        pred_masked = mu * mask + masked_image * (1 - mask)
        
        return mu, pred_masked, sigma

