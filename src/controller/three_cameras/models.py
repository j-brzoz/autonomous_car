import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_shape=(3, 192, 384), latent_dim=256):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.input_shape = input_shape

        # Encoder
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(16),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(512)
        )

        self.flatten = nn.Flatten()
        self.flattened_dim = 512 * 3 * 6

        self.fc_mu = nn.Linear(self.flattened_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_dim, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flattened_dim)

        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(256),
        
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),
        
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64),
        
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(32),
        
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(16),
        
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder_cnn(x)
        h = self.flatten(h)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, 512, 3, 6)
        return self.decoder_cnn(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

class MLP_classifier(nn.Module):
    def __init__(self, input_shape=256*3):
        super(MLP_classifier, self).__init__()

        self.input_shape = input_shape
        
        self.cls = nn.Sequential(
            nn.Linear(self.input_shape, 256),
            nn.Tanh(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Dropout(0.2),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.cls(x)

class MLP_regressor(nn.Module):
    def __init__(self, input_shape=256*3):
        super(MLP_regressor, self).__init__()

        self.input_shape = input_shape
        
        self.cls = nn.Sequential(
            nn.Linear(self.input_shape, 256),
            nn.Tanh(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Dropout(0.2),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.cls(x)