import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def create_dataloader(data_dir, image_size, batch_size, loader_workers):
    dataset = dset.ImageFolder(root=data_dir,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=loader_workers)

#smooth the plot of the losses with moving average
def moving_average(data, window_size):
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')

def plot_losses(G_losses, D_losses, smooth_amt=5, output_path='loss_plot.png'):
    g_smooth = moving_average(G_losses, smooth_amt)
    d_smooth = moving_average(D_losses, smooth_amt)
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_smooth,label="Generator")
    plt.plot(d_smooth,label="Discriminator")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()

    # save the training loss plot to output file to view and help debug
    plt.savefig(output_path)

def save_output_img(dataloader, img_list, device, image_size, output_path='output_images.png'):
    real_batch = next(iter(dataloader))

    # Plot some real images
    plt.figure(figsize=(25, 25))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:image_size], padding=5, normalize=True).cpu(),
                            (1, 2, 0)))

    # Plot the generated images saved from the final epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))

    # save to output file to view
    plt.savefig(output_path)