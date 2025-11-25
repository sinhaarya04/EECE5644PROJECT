"""
Conditional Neural Process (CNP) implementation for image inpainting
Based on: "Neural Processes" by Garnelo et al. (2018)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNP(nn.Module):
    """
    Conditional Neural Process for image inpainting.
    
    Architecture:
    - Encoder: Processes context points (x, y, RGB) -> embeddings
    - Aggregator: Mean pooling of context embeddings
    - Decoder: Predicts RGB values for target locations
    """
    
    def __init__(
        self,
        input_dim=5,  # (x, y, r, g, b)
        hidden_dim=128,
        encoder_layers=4,
        decoder_layers=3,
        output_dim=3,  # RGB
        use_attention=False
    ):
        super(CNP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_attention = use_attention
        
        # Encoder: processes context points
        encoder_layers_list = []
        encoder_layers_list.append(nn.Linear(input_dim, hidden_dim))
        encoder_layers_list.append(nn.ReLU())
        
        for _ in range(encoder_layers - 1):
            encoder_layers_list.append(nn.Linear(hidden_dim, hidden_dim))
            encoder_layers_list.append(nn.ReLU())
        
        self.encoder = nn.Sequential(*encoder_layers_list)
        
        # Optional attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                batch_first=True
            )
            # Projection layer for target locations to query embeddings
            self.query_proj = nn.Linear(2, hidden_dim)
        
        # Decoder: predicts RGB for target locations
        # Input: target location (x, y) + aggregated context
        decoder_input_dim = 2 + hidden_dim  # (x, y) + context embedding
        
        decoder_layers_list = []
        decoder_layers_list.append(nn.Linear(decoder_input_dim, hidden_dim))
        decoder_layers_list.append(nn.ReLU())
        
        for _ in range(decoder_layers - 1):
            decoder_layers_list.append(nn.Linear(hidden_dim, hidden_dim))
            decoder_layers_list.append(nn.ReLU())
        
        # Output layer: predict RGB values
        decoder_layers_list.append(nn.Linear(hidden_dim, output_dim))
        decoder_layers_list.append(nn.Sigmoid())  # Ensure output in [0, 1]
        
        self.decoder = nn.Sequential(*decoder_layers_list)
    
    def forward(self, context_x, context_y, target_x):
        """
        Forward pass through CNP.
        
        Args:
            context_x: Context input points [B, N_c, input_dim]
                      where input_dim includes (x, y, r, g, b)
            context_y: Context output values [B, N_c, output_dim] (RGB values)
            target_x: Target input points [B, N_t, 2] (x, y coordinates)
        
        Returns:
            pred_y: Predicted RGB values for target points [B, N_t, output_dim]
            mu: Mean predictions (same as pred_y for CNP)
            sigma: Uncertainty estimates (optional, can be added)
        """
        batch_size = context_x.shape[0]
        n_context = context_x.shape[1]
        n_target = target_x.shape[1]
        
        # Encode context points
        # context_x: [B, N_c, input_dim] -> [B, N_c, hidden_dim]
        context_embeddings = self.encoder(context_x)
        
        # Aggregate context embeddings (mean pooling)
        if self.use_attention:
            # Use attention to aggregate context
            # Query: target locations projected to embeddings
            query = self.query_proj(target_x)  # [B, N_t, hidden_dim]
            
            # Apply attention: query from targets, key/value from context
            aggregated, _ = self.attention(query, context_embeddings, context_embeddings)
            # aggregated: [B, N_t, hidden_dim]
        else:
            # Mean pooling: [B, N_c, hidden_dim] -> [B, hidden_dim]
            aggregated = context_embeddings.mean(dim=1)  # [B, hidden_dim]
            # Expand to match target points: [B, hidden_dim] -> [B, N_t, hidden_dim]
            aggregated = aggregated.unsqueeze(1).expand(-1, n_target, -1)
        
        # Decode: concatenate target locations with aggregated context
        # target_x: [B, N_t, 2], aggregated: [B, N_t, hidden_dim]
        decoder_input = torch.cat([target_x, aggregated], dim=-1)  # [B, N_t, 2+hidden_dim]
        
        # Predict RGB values
        pred_y = self.decoder(decoder_input)  # [B, N_t, output_dim]
        
        return pred_y, pred_y, None  # (pred_y, mu, sigma) for compatibility


class CNPWithUncertainty(nn.Module):
    """
    CNP variant that also predicts uncertainty (variance).
    """
    
    def __init__(
        self,
        input_dim=5,
        hidden_dim=128,
        encoder_layers=4,
        decoder_layers=3,
        output_dim=3,
        use_attention=False
    ):
        super(CNPWithUncertainty, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention
        
        # Same encoder as CNP
        encoder_layers_list = []
        encoder_layers_list.append(nn.Linear(input_dim, hidden_dim))
        encoder_layers_list.append(nn.ReLU())
        
        for _ in range(encoder_layers - 1):
            encoder_layers_list.append(nn.Linear(hidden_dim, hidden_dim))
            encoder_layers_list.append(nn.ReLU())
        
        self.encoder = nn.Sequential(*encoder_layers_list)
        
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                batch_first=True
            )
            # Projection layer for target locations to query embeddings
            self.query_proj = nn.Linear(2, hidden_dim)
        
        # Decoder with two heads: mean and variance
        decoder_input_dim = 2 + hidden_dim
        
        # Shared layers
        shared_layers = []
        shared_layers.append(nn.Linear(decoder_input_dim, hidden_dim))
        shared_layers.append(nn.ReLU())
        
        for _ in range(decoder_layers - 1):
            shared_layers.append(nn.Linear(hidden_dim, hidden_dim))
            shared_layers.append(nn.ReLU())
        
        self.shared_decoder = nn.Sequential(*shared_layers)
        
        # Mean head
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        
        # Variance head (predict log variance for numerical stability)
        self.var_head = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus()  # Ensures positive variance
        )
        
        self.use_attention = use_attention
    
    def forward(self, context_x, context_y, target_x):
        """
        Forward pass that returns both mean and variance.
        
        Returns:
            pred_y: Predicted RGB values [B, N_t, output_dim]
            mu: Mean predictions [B, N_t, output_dim]
            sigma: Standard deviation [B, N_t, output_dim]
        """
        batch_size = context_x.shape[0]
        n_context = context_x.shape[1]
        n_target = target_x.shape[1]
        
        # Encode context
        context_embeddings = self.encoder(context_x)
        
        # Aggregate
        if self.use_attention:
            query = self.query_proj(target_x)  # [B, N_t, hidden_dim]
            aggregated, _ = self.attention(query, context_embeddings, context_embeddings)
        else:
            aggregated = context_embeddings.mean(dim=1)
            aggregated = aggregated.unsqueeze(1).expand(-1, n_target, -1)
        
        # Decode
        decoder_input = torch.cat([target_x, aggregated], dim=-1)
        shared_features = self.shared_decoder(decoder_input)
        
        # Predict mean and variance
        mu = self.mean_head(shared_features)
        log_var = self.var_head(shared_features)
        sigma = torch.sqrt(log_var + 1e-6)  # Add small epsilon for numerical stability
        
        return mu, mu, sigma

