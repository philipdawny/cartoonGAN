# Discriminator code
import torch.nn as nn
class Discriminator(nn.Module):
    def __init__(self, ngpu, dim_args={}):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        disc_dim = dim_args["disc_dim"]
        num_channels = dim_args["num_channels"]
        self.main = nn.Sequential(
            nn.Conv2d(num_channels, disc_dim, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(disc_dim),  # Add batch normalization
            nn.LeakyReLU(0.2, inplace=True),  # Use LeakyReLU with a slope of 0.2
            nn.Conv2d(disc_dim, disc_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(disc_dim * 2),  # Add batch normalization
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(disc_dim * 2, disc_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(disc_dim * 4),  # Add batch normalization
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(disc_dim * 4, disc_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(disc_dim * 8),  # Add batch normalization
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(disc_dim * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  # Use sigmoid activation at the end
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)
