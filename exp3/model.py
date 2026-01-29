# Imports section
import numpy as np

import torch
import torch.nn as nn

# Seed definition
seed = 123
np.random.seed(seed)
_ = torch.manual_seed(seed)
_ = torch.cuda.manual_seed(seed)

# Work on GPU if it is available in the machine, otherwise run on CPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

# Encoder
# Residual block for Qcurve
class QResBlock(nn.Module):
    """
    Residual block with 3 conv layers:
    input
    -> Conv1d(kernel = 3) -> LeakyReLU
    -> Conv1d(kernel = 3) -> LeakyReLU
    -> Conv1d(kernel = 3) -> LeakyReLU
    -> skip connection -> LeakyReLU
    -> output
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            ) -> None:
        
        super(QResBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.middle_channels = self.out_channels//2
    
        self.conv1 = nn.Conv1d(in_channels=self.in_channels, out_channels=self.middle_channels, kernel_size=1, stride=1, dilation=1, padding=0, bias=True)
        self.batchNorm1 = nn.BatchNorm1d(num_features=self.middle_channels)
        self.act1 = nn.LeakyReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(in_channels=self.middle_channels, out_channels=self.middle_channels, kernel_size=3, stride=1, dilation=1, padding=1, bias=True)
        self.batchNorm2 = nn.BatchNorm1d(num_features=self.middle_channels)
        self.act2 = nn.LeakyReLU(inplace=True)
        
        self.conv3 = nn.Conv1d(in_channels=self.middle_channels, out_channels=self.out_channels, kernel_size=1, stride=1, dilation=1, padding=0, bias=True)
        self.batchNorm3 = nn.BatchNorm1d(num_features=self.out_channels)
        self.act3 = nn.LeakyReLU(inplace=True)
        
        self.act4 = nn.LeakyReLU(inplace=True)

        self.skip_conv = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=1, dilation=1, padding=0, bias=True)


    def forward(self, x):
        #print(x.shape) # [5, 1, 512]

        skip = self.skip_conv(x)

        out = self.conv1(x)
        out = self.act1(out)
        
        #print(out.shape) # [5, 32, 512]
        out = self.conv2(out)
        out = self.act2(out)

        #print(out.shape) # [5, 32, 512]
        out = self.conv3(out)
        out = self.act3(out)

        #print(out.shape) # [5, 64, 512]
        out = out + skip  # residual connection
        out = self.act4(out)
        return out
    
# Encoder
# Pitch curve encoder
class Qcurve(nn.Module):
    """
    Qcurve encodes the Frp from input curve into latent pitch vector Zrp
    CNN composed by 7 residual blocks (QResBlock)
    """
    def __init__(
            self,
            channels: int,
            latent_dim: int,
            frp_len: int,
            n_layers: int,
            ) -> None:

        super(Qcurve, self).__init__()

        self.base_channels = channels
        self.latent_dim = latent_dim
        self.frp_len = frp_len
        self.n_layers = n_layers

        self.curve_encoder = nn.Sequential()
        
        current_channels = 1

        for i in range(self.n_layers):
            # Add 7 layers of residual blocks to the curve_encoder
            out_channels = self.base_channels * 2**i # out_channels gets bigger with each layer
            resBlock = QResBlock(in_channels=current_channels, out_channels=out_channels)
            self.curve_encoder.add_module(f'layer{i}', resBlock)
            current_channels = out_channels
        
        self.mu = nn.Conv1d(in_channels=current_channels, out_channels=self.latent_dim, kernel_size=self.frp_len)
        self.logvar = nn.Conv1d(in_channels=current_channels, out_channels=self.latent_dim, kernel_size=self.frp_len)
    
    def reparametrize(
            self,
            mu:torch.Tensor,
            logvar:torch.Tensor,
            ) -> torch.Tensor:
        # Given mean and logvar returns z
        # reparameterization trick: instead of sampling from Q(z|X), sample epsilon = N(0,I)
        # mu, logvar: mean and log of variance of Q(z|X)

        # The factor 1/2 in the exponent ensures that the distribution has unit variance
        std = torch.exp(0.5 * logvar)
        # Random sample
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x: torch.Tensor):
        #print(x.shape) # [5, 1, 512]
        out = self.curve_encoder(x)
        #print(out.shape) # [5, 4096, 512]

        x_mu = self.mu(out).squeeze(-1)
        #print(x_mu.shape) # [5, 85]
        x_logvar = self.logvar(out).squeeze(-1)
        #print(x_logvar.shape) # [5, 85]
        z = self.reparametrize(x_mu, x_logvar)
        #print(z.shape) # [5, 85]

        return z, x_mu, x_logvar

# Encoder
# Context encoder
class Qcontext(nn.Module):
    """
    Qcontext encodes the past context bar Mprev into the latent context vector Zc
    Pipeline:
    - 1 embedding layer of size 128
    - 2 layers of bidirectional GRUs (2048 units each)
    - 1 dense layer
    """

    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim_gru: int,
                 n_layers_gru: int,
                 latent_dim: int,
                 ) -> None:
        
        super(Qcontext, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim_gru = hidden_dim_gru
        self.n_layers_gru = n_layers_gru
        self.latent_dim = latent_dim

        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim)
        self.gru = nn.GRU(input_size=self.embedding_dim, hidden_size=self.hidden_dim_gru, num_layers=self.n_layers_gru, batch_first=True, bidirectional=True)

        self.mu = nn.Linear(in_features=self.hidden_dim_gru*2, out_features=self.latent_dim)
        self.logvar = nn.Linear(in_features=self.hidden_dim_gru*2, out_features=self.latent_dim)

    def reparametrize(
            self,
            mu:torch.Tensor,
            logvar:torch.Tensor,
            ) -> torch.Tensor:
        # Given mean and logvar returns z
        # reparameterization trick: instead of sampling from Q(z|X), sample epsilon = N(0,I)
        # mu, logvar: mean and log of variance of Q(z|X)

        # The factor 1/2 in the exponent ensures that the distribution has unit variance
        std = torch.exp(0.5 * logvar)
        # Random sample
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x: torch.Tensor):
        #print(x.shape) # [8, 1, 48]
        embed = self.embedding(x).squeeze(1)
        #print(embed.shape) # [8, 48, 128]

        out, _ = self.gru(embed)
        #print(out.shape) # [8, 48, 4096]

        h = out[:, -1, :] # take last timestep as context summary
        #print(h.shape) # [8, 4096]

        h_mu = self.mu(h)
        #print(h_mu.shape) # [8, 85]
        h_logvar = self.logvar(h)
        #print(h_logvar.shape) # [8, 85]
        z = self.reparametrize(h_mu, h_logvar)
        #print(z.shape) # [8, 85]
    
        return z, h_mu, h_logvar

# Decoder
# Pitch decoder
class Prp(nn.Module):
    """
    Prp decodes the Zrp vector into the relative pitch tokens Trp
    Has 2 layers of unidirectional GRUs (2048 units each)
    Uses teacher forcing for training
    """

    def __init__(self,
                 latent_dim: int,
                 vocab_size_Trp: int,
                 seq_len: int,
                 hidden_dim_decoder: int,
                 n_layers: int,
                 ) -> None:
        
        super(Prp, self).__init__()

        self.latent_dim = latent_dim
        self.vocab_size_Trp = vocab_size_Trp
        self.seq_len = seq_len
        self.hidden_dim_decoder = hidden_dim_decoder
        self.n_layers = n_layers

        self.gru = nn.GRU(input_size=self.latent_dim, hidden_size=self.hidden_dim_decoder, num_layers=self.n_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(in_features=self.hidden_dim_decoder, out_features=self.vocab_size_Trp)

    def forward(self, z: torch.Tensor):
        #print(z.shape) # [8, 85]

        B = z.size(0)

        # Reshape z so its (B, seq_len, input_size)
        z = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        #print(z.shape) # [8, 24, 85]

        # Initialize hidden state -> input for gru
        h0 = torch.zeros(self.n_layers, B, self.hidden_dim_decoder, device=z.device)

        out, _ = self.gru(z, h0) # output: output tensor, final hidden state tensor
        #print(out.shape) # [8, 24, 2048]
        tokens = self.fc(out)
        #print(tokens.shape) # [8, 24, 7]

        return tokens

# Decoder
# MIDI decoder
class Pmidi(nn.Module):
    """
    Pmidi decodes Trp + Zc into Tmidi
    Has 2 layers of unidirectional GRUs (2048 units each)
    Uses cross-entropy losses during training
    """

    def __init__(self,
                 latent_dim: int,
                 vocab_size_Trp: int,
                 seq_len: int,
                 hidden_dim_decoder: int,
                 n_layers: int,
                 vocab_size: int,
                 ) -> None:
        
        super(Pmidi, self).__init__()

        self.latent_dim = latent_dim
        self.vocab_size_Trp = vocab_size_Trp
        self.seq_len = seq_len
        self.hidden_dim_decoder = hidden_dim_decoder
        self.n_layers = n_layers
        self.vocab_size = vocab_size

        self.gru = nn.GRU(input_size=self.latent_dim + self.vocab_size_Trp, hidden_size=self.hidden_dim_decoder, num_layers=self.n_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(in_features=self.hidden_dim_decoder, out_features=self.vocab_size)
    
    def forward(self, trp, zc: torch.Tensor):
        #print(trp.shape) # [8, 24, 7]
        # trp: [B, seq_len, vocab_size_trp]

        #print(zc.shape) # [8, 85]
        B = zc.size(0)
        # zc: [B, latent_dim] -> expand to seq_len
        zc = zc.unsqueeze(1).repeat(1, self.seq_len, 1)
        #print(zc.shape) # [8, 24, 85]

        # concat trp+zc : [B, seq_len, vocab_size_trp+latent_dim]
        xmidi = torch.cat([zc, trp], dim=-1)
        #print(xmidi.shape) # [8, 24, 92]

        # Initialize hidden state -> input for gru
        h0 = torch.zeros(self.n_layers, B, self.hidden_dim_decoder, device=xmidi.device)

        out, _ = self.gru(xmidi, h0) # output: output tensor, final hidden state tensor
        #print(out.shape) # [8, 24, 2048]
        tokens = self.fc(out)
        #print(tokens.shape) # [8, 24, 130]
        
        return tokens

# Decoder
# Chroma decoder
class Pchroma(nn.Module):
    """
    Pmidi decodes Zc inot Vchroma
    Has a MLP block
    """

    def __init__(self,
                 latent_dim: int,
                 chroma_dim: int,
                 ) -> None:
        
        super(Pchroma, self).__init__()

        self.latent_dim = latent_dim
        self.chroma_dim = chroma_dim

        self.linear = nn.Linear(in_features=self.latent_dim, out_features=self.chroma_dim)

    def forward(self, zc):
        vchroma = self.linear(zc)
        #print(vchroma.shape) # [8, 12]
       
        return vchroma

# Model
class VariationalAutoencoder(nn.Module):
    """
    Model composed by:
    - Encoders: Qcurve, Qcontext
    - Decoders: Prp, Pmidi
    """
    def __init__(self,
                 channels: int,
                 latent_dim: int,
                 vocab_size: int,
                 seq_len: int,
                 frp_len: int,
                 n_layers_encoder: int,
                 embedding_dim: int,
                 n_layers_gru: int,
                 hidden_dim_gru: int,
                 vocab_size_Trp: int,
                 chroma_dim: int,
                 ) -> None:
        
        super(VariationalAutoencoder, self).__init__()

        self.channels = channels
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.frp_len = frp_len
        self.n_layers_encoder = n_layers_encoder
        self.embedding_dim = embedding_dim
        self.n_layer_gru = n_layers_gru
        self.hidden_dim_gru = hidden_dim_gru
        self.vocab_size_Trp = vocab_size_Trp
        self.chroma_dim = chroma_dim

        self.encoder_curve = Qcurve(channels=self.channels, latent_dim=self.latent_dim, frp_len=self.frp_len, n_layers=self.n_layers_encoder)
        self.encoder_context = Qcontext(vocab_size=self.vocab_size, embedding_dim=self.embedding_dim, hidden_dim_gru=self.hidden_dim_gru, n_layers_gru=self.n_layer_gru, latent_dim=self.latent_dim)
        self.decoder_pitch = Prp(latent_dim=self.latent_dim, vocab_size_Trp=self.vocab_size_Trp, seq_len=self.seq_len, hidden_dim_decoder=self.hidden_dim_gru, n_layers=self.n_layer_gru)
        self.decoder_midi = Pmidi(latent_dim=self.latent_dim, vocab_size_Trp= self.vocab_size_Trp, seq_len=self.seq_len, hidden_dim_decoder=self.hidden_dim_gru, n_layers=self.n_layer_gru, vocab_size=self.vocab_size)
        self.decoder_chroma = Pchroma(latent_dim=self.latent_dim, chroma_dim=self.chroma_dim)

    def forward(self, x_curve, x_context: torch.Tensor):
        #print(x_curve.shape) # [8, 1, 512]
        #print(x_context.shape) # [8, 1, 48]

        # Go through encoders
        zrp, zrp_mu, zrp_logvar = self.encoder_curve(x_curve)
        #print(zrp.shape) # [8, 85]
        zc, zc_mu, zc_logvar = self.encoder_context(x_context)
        #print(zc.shape) # [8, 85]

        # Go trhough decoders
        trp = self.decoder_pitch(zrp)
        #print(trp.shape) # [8, 24, 7]
        tmidi = self.decoder_midi(trp, zc)
        vchroma = self.decoder_chroma(zc)

        return tmidi, trp, vchroma, zrp_mu, zrp_logvar, zc_mu, zc_logvar