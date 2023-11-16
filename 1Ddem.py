import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import grad
np.random.seed(2023)
torch.manual_seed(2023)

Length = 1.0
x0 = -1
N = 1000
test_size = 1000

Height = 1.0
Depth = 1.0
known_left_ux = 0
bc_left_penalty = 1.0

known_right_tx = 0
bc_right_penalty = 1.0
dev = torch.device('cpu')

config = {
    'lr': 0.1,
    'epochs': 100
}

times = {}
def exact(x):
    return 1./135*(68 + 105*x - 40*x**3 + 3*x**5)

def p_energy(u, x):
    # eps = grad(u, x, torch.ones(len(x), 1), create_graph=True, retain_graph=True)[0]
    eps = grad(u, x, torch.ones(x.shape), create_graph=True, retain_graph=True)[0]
    energy = pow(1 + eps, 3/2) - 3/2*eps - 1
    return energy


class NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NN, self).__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = self.l2(x)
        return x

def train_model(data, max_it=20, j=-1):
    j += 1
    model = NN(1, 10, 1)
    x = torch.from_numpy(data).float()
    # x = x.to(dev)
    x.requires_grad_(True)

    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'])
    loss = []
    start_time = time.time()
    for i in range(max_it):
        u_pred = (x + 1) * model(x)
        boundary = u_pred*x
        energy = p_energy(u_pred, x)
        energy_loss = (Length - x0)*penalty(energy) - (Length - x0)*penalty(boundary)
        if np.isnan(energy_loss.detach().numpy()):
            # eps = grad(u_pred, x, torch.ones(x.shape))[0]
            # plt.subplot(121)
            # plt.plot(x.detach().numpy(), u_pred.detach().numpy())
            # plt.subplot(122)
            # plt.plot(x.detach().numpy(), eps.detach().numpy())
            # plt.show(); exit()
            print(j)
            pass
        else:
            loss = energy_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print(f'Iter: {i+1:d}, Loss: {energy_loss.item():.5f}')

    elapsed = time.time() - start_time
    return model, elapsed, j

def penalty(input):
    return torch.sum(input) / input.data.nelement()

def energy_pen(input):
    return torch.mean(input)


def evaluate_model(model, test_data):
    x = torch.from_numpy(test_data).float()
    x.requires_grad_(True)
    u_pred = (x + 1) * model(x)
    # print(u_pred.shape)
    # print(x.shape)
    # exit()
    energy = p_energy(u_pred, x)
    e_loss = (Length-x0)*(penalty(energy) - penalty(u_pred*x))
    print(f'evaluation_loss: {e_loss}')
    u_pred = u_pred.detach().numpy()
    e_loss = e_loss.detach().numpy()
    return u_pred, e_loss


if __name__ == '__main__':
    x = np.linspace(-1, 1, 301)
    y = exact(x)
    # dy = np.gradient(y)
    dy = 1./135*(105 - 120*x**2 + 15*x*4)
    
    dx = x[1] - x[0]
    energy = (1+dy)**(3/2) - 3/2*dy - 1
    print(np.trapz(dy - y*x, x, dx=dx))
    # plt.plot(x, y)
    # plt.plot(x, energy)
    # plt.show(); exit()

    test_set = np.array([np.linspace(x0, Length, test_size)]).T
    times = {}
    losses = {}
    # for N in [621]:
    # for N in np.linspace(10, 10000, 50, dtype=int):
    j=0
    for N in np.logspace(2, 5, 5, dtype=int):
        domain = np.array([np.linspace(x0, Length, N)]).T
        # train
        # j=0
        model, elapsed, j = train_model(domain, max_it=config['epochs'], j=j)
        times[N] = elapsed
        # predict
        u_pred, loss = evaluate_model(model, test_set)
        losses[N] = loss

    l1 = sorted(times.items())
    a,b = zip(*l1)
    plt.figure()
    plt.loglog(a, b)
    # plt.show(); exit()

    # l1 = sorted(losses.items())
    # a,b = zip(*l1)
    # plt.figure()
    # plt.plot(a, b)

    # plt.show(); exit()
    # test_set = np.sort(np.random.uniform(x0, Length, size=(test_size, 1)), axis=0)
    plt.figure()
    plt.plot(test_set, exact(test_set))
    plt.plot(test_set, u_pred)
    plt.legend(['exact', 'dem'])
    plt.show()
    # print(test_set.shape, u_pred.shape)