# Generator Code
import torch.nn as nn
class Generator(nn.Module):
    def __init__(self, ngpu, dim_args={}):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        random_dim = dim_args["random_dim"]
        gen_dim = dim_args["gen_dim"]
        num_channels = dim_args["num_channels"]
        self.main = nn.Sequential(
            nn.ConvTranspose2d(random_dim, gen_dim * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(gen_dim * 8),  # Add batch normalization
            nn.ReLU(True),
            nn.ConvTranspose2d(gen_dim * 8, gen_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_dim * 4),  # Add batch normalization
            nn.ReLU(True),
            nn.ConvTranspose2d(gen_dim * 4, gen_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_dim * 2),  # Add batch normalization
            nn.ReLU(True),
            nn.ConvTranspose2d(gen_dim * 2, gen_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_dim),  # Add batch normalization
            nn.ReLU(True),
            nn.ConvTranspose2d(gen_dim, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output
