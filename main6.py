import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Generator Network
G = nn.Sequential(
    nn.Linear(20, 128),
    nn.ReLU(),
    nn.Linear(128, 784),
    nn.Sigmoid()
)

# Discriminator Network
D = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid()
)

# Loss and Optimizers
loss_fn = nn.BCELoss()
opt_G = torch.optim.Adam(G.parameters(), lr=0.0002)
opt_D = torch.optim.Adam(D.parameters(), lr=0.0002)

# Training Loop
for epoch in range(100):
    # Generate fake data
    z = torch.randn(32, 20)
    fake_data = G(z)

    # Real data (random for this example)
    real_data = torch.rand(32, 784)

    # Labels
    real_labels = torch.ones(32, 1)
    fake_labels = torch.zeros(32, 1)

    # Train Discriminator
    D_real = D(real_data)
    D_fake = D(fake_data.detach())
    loss_D = loss_fn(D_real, real_labels) + loss_fn(D_fake, fake_labels)

    opt_D.zero_grad()
    loss_D.backward()
    opt_D.step()

    # Train Generator
    D_fake = D(fake_data)
    loss_G = loss_fn(D_fake, real_labels)

    opt_G.zero_grad()
    loss_G.backward()
    opt_G.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss_D = {loss_D.item():.4f}, Loss_G = {loss_G.item():.4f}")

# Visualize one generated sample
sample = G(torch.randn(1, 20)).view(28, 28).detach()
plt.imshow(sample, cmap='gray')
plt.title("Generated Image")
plt.show()