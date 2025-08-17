import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generator network
class Generator(nn.Module):
    def __init__(self, noise_dim, text_embed_dim, img_size):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.fc1 = nn.Linear(noise_dim + text_embed_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, img_size * img_size * 3)  # 3 channels (RGB)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, noise, text_embedding):
        x = torch.cat((noise, text_embedding), dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        x = x.view(-1, 3, self.img_size, self.img_size)
        return x


# Discriminator network
class Discriminator(nn.Module):
    def __init__(self, img_size, text_embed_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(img_size * img_size * 3 + text_embed_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, img, text_embedding):
        img = img.view(img.size(0), -1)  # Flatten image
        x = torch.cat((img, text_embedding), dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


# Hyperparameters
noise_dim = 100
text_embed_dim = 128
img_size = 64
batch_size = 32
lr = 0.0002
num_epochs = 100

# Create generator and discriminator
G = Generator(noise_dim, text_embed_dim, img_size).to(device)
D = Discriminator(img_size, text_embed_dim).to(device)

# Loss and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=lr)
optimizer_D = optim.Adam(D.parameters(), lr=lr)

# Function to generate random noise and text embeddings
def generate_noise(batch_size, noise_dim):
    return torch.randn(batch_size, noise_dim).to(device)

def generate_text_embeddings(batch_size, text_embed_dim):
    return torch.randn(batch_size, text_embed_dim).to(device)  # Fake embeddings


# Training loop
for epoch in range(num_epochs):
    for _ in range(batch_size):  # Simulating mini-batches
        # Real images and embeddings (simulated)
        real_imgs = torch.randn(batch_size, 3, img_size, img_size).to(device)
        real_text_embeddings = generate_text_embeddings(batch_size, text_embed_dim)
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Generate fake images
        noise = generate_noise(batch_size, noise_dim)
        fake_text_embeddings = generate_text_embeddings(batch_size, text_embed_dim)
        fake_imgs = G(noise, fake_text_embeddings)

        # Train Discriminator
        outputs_real = D(real_imgs, real_text_embeddings)
        d_loss_real = criterion(outputs_real, real_labels)

        outputs_fake = D(fake_imgs.detach(), fake_text_embeddings)
        d_loss_fake = criterion(outputs_fake, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        outputs_fake = D(fake_imgs, fake_text_embeddings)
        g_loss = criterion(outputs_fake, real_labels)
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}] | d_loss: {d_loss.item():.4f} | g_loss: {g_loss.item():.4f}")

    # Save generated images every 10 epochs
    if (epoch + 1) % 10 == 0:
        save_image(fake_imgs.data[:25], f'generated_images_{epoch + 1}.png', nrow=5, normalize=True)

# Save the trained models
torch.save(G.state_dict(), 'generator.pth')
torch.save(D.state_dict(), 'discriminator.pth')
