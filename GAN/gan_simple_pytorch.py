#-*- coding:utf-8 -*-

# reference: https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd 
import torch.optim as optim
from torch.autograd import Variable


def get_distribution_sampler(mu, sigma):
    """Return the original data distribution: normal distribution
    
    Args:
        mu (TYPE): mean value
        sigma (TYPE): standard deviation
    """
    return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1, n)))

def get_generator_input_sampler():
    """Return the input distribution to the generator: uniform distribution
    """
    return lambda m, n: torch.rand(m, n)

class Generator(nn.Module):
    """Generator: Feedforward Neural Network
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.elu(self.map1(x))
        x = F.sigmoid(self.map2(x))
        return self.map3(x)

class Discriminator(nn.Module):
    """Discriminator: Feedforward Neural Network
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.elu(self.map1(x))
        x = F.elu(self.map2(x))
        return F.sigmoid(self.map3(x))

def extract(v): 
    return v.data.storage().tolist()

def stats(d): 
    return [np.mean(d), np.std(d)]

def train():
    # parameter definition
    d_input_size = 100 
    g_input_size = 100
    minibatch_size = 100
    num_epochs = 30000
    d_steps = 1 
    g_steps = 1 
    print_interval = 200
    # data and network definition
    d_sampler = get_distribution_sampler(mu=4, sigma=1.25)
    g_sampler = get_generator_input_sampler()
    d_network = Discriminator(input_size=d_input_size, hidden_size=50, output_size=1)
    g_network = Generator(input_size=g_input_size, hidden_size=50, output_size=1)

    # training parameters definition 
    criterion = nn.BCELoss() # Binary Cross Entropy: http://pytorch.org/docs/nn.html#bceloss
    d_optimizer = optim.Adam(d_network.parameters(), lr=2e-4, betas=(0.9, 0.999))
    g_optimizer = optim.Adam(g_network.parameters(), lr=2e-4, betas=(0.9, 0.999))


    for epoch in range(num_epochs):
        # train the discriminator 
        for d_index in range(d_steps):
            d_network.zero_grad()

            ## train discriminator on real
            d_real_data = Variable(d_sampler(d_input_size))
            d_real_decision =  d_network(d_real_data)
            d_real_error = criterion(d_real_decision, Variable(torch.ones(1)))
            d_real_error.backward()

            ## train discriminator on fake
            d_gen_input = Variable(g_sampler(minibatch_size, g_input_size))
            d_fake_data = g_network(d_gen_input).detach() # detach to avoid training generator on these labels
            d_fake_decision = d_network(d_fake_data.t()) # why transpose
            d_fake_error = criterion(d_fake_decision, Variable(torch.zeros(1)))
            d_fake_error.backward()
            
            d_optimizer.step()

        # train the generator
        for g_index in range(g_steps):
            g_network.zero_grad()
            gen_input = Variable(g_sampler(minibatch_size, g_input_size))
            g_fake_data = g_network(gen_input)
            dg_fake_decision = d_network(g_fake_data.t())
            g_error = criterion(dg_fake_decision, Variable(torch.ones(1)))
            g_error.backward()
            g_optimizer.step()

        if epoch % print_interval == 0:
            print 'Epoch: %d:' % epoch
            print 'Loss: discriminator: %.2f/%.2f, generator: %.2f' % (
                extract(d_real_error)[0], extract(d_fake_error)[0], extract(g_error)[0])
            print 'Data: real: (%.2f, %.2f), fake: (%.2f, %.2f)' % tuple(
                stats(extract(d_real_data)) + stats(extract(d_fake_data))
            )

if __name__ == '__main__':
    train()
