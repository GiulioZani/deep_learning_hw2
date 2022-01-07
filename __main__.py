from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
from .ml_utils.misc import CurrentDir, get_settings
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from torch.optim.lr_scheduler import StepLR
from sklearn.decomposition import PCA

curdir = CurrentDir(__file__)
settings = get_settings(curdir)


class ConvVAE(nn.Module):
    def __init__(self):
        super(ConvVAE, self).__init__()
        self.conv_channels = 32
        self.n_elements = 21632
        self.conv = nn.Conv2d(1, self.conv_channels, 3)
        self.fc21 = nn.Linear(self.n_elements, 20)
        self.fc22 = nn.Linear(self.n_elements, 20)
        self.fc3 = nn.Linear(20, self.n_elements)
        self.t_conv = nn.ConvTranspose2d(self.conv_channels, 1, 3)

    def encode(self, x):
        x = F.relu(self.conv(x))
        x = torch.flatten(x, 1)
        return self.fc21(x), self.fc22(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        x = h3.reshape(-1, self.conv_channels, 26, 26)
        x = self.t_conv(x)
        return torch.sigmoid(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VAEManager:
    def __init__(self, args):
        if not os.path.exists('results'):
            os.mkdir('results')
        self.__dict__.update(args)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        kwargs = {'num_workers': 1, 'pin_memory': True}
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./__data__',
                           train=True,
                           download=True,
                           transform=transforms.ToTensor()),
            batch_size=self.batch_size,
            shuffle=True,
            **kwargs)
        self.test_dataset = datasets.MNIST('./__data__',
                                           train=False,
                                           transform=transforms.ToTensor())
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            **kwargs)
        self.model = ConvVAE().to(self.device)
        print(self.model)
        if os.path.exists(curdir(
                self.model_file_name)) and not self.force_reload:
            self.model.load_state_dict(torch.load(curdir(
                self.model_file_name)))

        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=1e-3,
                                    weight_decay=self.weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=self.gamma)

    @staticmethod
    def loss_function(recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x.view(-1, 784),
                                     x.view(-1, 784),
                                     reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def train_single_epoch(self, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(data)
            loss = VAEManager.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            self.scheduler.step()
            if batch_idx % self.log_interval == 0:
                print(
                    f'Train Epoch: {epoch}',
                    f'[{batch_idx * len(data)}/{len(self.train_loader.dataset)}',
                    f'({round(100. * batch_idx / len(self.train_loader))}%)]\t',
                    f'Loss: {round(loss.item() / len(data), 2)}')
        # print(f'====> Epoch: {epoch}',
        #      f' Average loss: {train_loss / len(self.train_loader.dataset)}')

    def test(self):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(self.test_loader):
                data = data.to(self.device)
                recon_batch, mu, logvar = self.model(data)
                test_loss += VAEManager.loss_function(recon_batch, data, mu,
                                                      logvar).item()
                if i == 0:
                    n = min(data.size(0), 8)
                    comparison = torch.cat(
                        (data[:n], recon_batch.view(self.batch_size, 1, 28,
                                                    28)[:n]))
                    # save_image(comparison.cpu(),
                    #           f'results/reconstruction_{epoch}.png',
                    #           nrow=n)

        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        return test_loss

    def train(self):
        for epoch in range(1, self.epochs + 1):
            self.train_single_epoch(epoch)
            if epoch * self.test_interval == 0:
                self.test(epoch)
            if epoch % self.save_img_interval == 0:
                with torch.no_grad():
                    sample = torch.randn(64, 20).to(self.device)
                    decoded_sample = self.model.decode(sample).cpu()
                    # save_image(decoded_sample.view(64, 1, 28, 28),
                    #           f'results/sample_{epoch}.png')

    def save(self):
        torch.save(self.model.cpu().state_dict(), self.model_file_name)

    def plot_latent_space(self):
        with torch.no_grad():
            cmap = plt.cm.rainbow
            norm = matplotlib.colors.Normalize(vmin=0, vmax=9)
            count = 0
            train_loader = torch.utils.data.DataLoader(datasets.MNIST(
                './__data__',
                train=True,
                download=True,
                transform=transforms.ToTensor()),
                                                       batch_size=5000)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            pca = PCA(n_components=3)
            for batch_idx, (data, labels) in enumerate(train_loader):
                data = data.to(self.device)
                mu, logvar = self.model.encode(data)
                z = self.model.reparameterize(mu, logvar)
                print(z.shape)
                """
                recon_batch, mu, logvar = self.model(data)
                loss = VAEManager.loss_function(recon_batch, data, mu, logvar)
                loss.backward()
                train_loss += loss.item()
                """
                labels = labels.numpy().tolist()
                out = pca.fit_transform(z)
                # import ipdb
                # ipdb.set_trace()
                for i, (x, y, z) in enumerate(out):
                    c = cmap(norm(labels[i]))
                    # ipdb.set_trace()
                    ax.scatter(x, y, z, color=c)
                break
            plt.show()

    def denoise(self):
        n = 10
        for i, (batch, _) in enumerate(self.test_loader):
            data = batch[:n]
            break
        noise_factor = 0.2
        noisy_data = data + noise_factor * np.random.normal(
            loc=0.0, scale=1.0, size=data.shape)

        noisy_data = torch.tensor(np.clip(noisy_data, 0., 1.)).float()
        print(noisy_data.dtype)
        print(noisy_data.shape)
        decoded_imgs = self.model(noisy_data.to(self.device))[0]
        plt.figure(figsize=(20, 4))

        for i in range(n):
            # display original
            ax = plt.subplot(2, n, i + 1)
            print(noisy_data.shape)
            print(decoded_imgs.shape)
            plt.imshow(noisy_data[i][0].reshape(28, 28).detach().cpu())
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(decoded_imgs[i][0].reshape(28, 28).detach().cpu())
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.savefig(curdir('imgs/denoise.png'))


def main():
    VAE_manager = VAEManager(settings)
    VAE_manager.test()
    # VAE_manager.plot_latent_space()
    # self.save()
    # VAE_manager.train()
    # VAE_manager.denoise()


if __name__ == "__main__":
    main()
