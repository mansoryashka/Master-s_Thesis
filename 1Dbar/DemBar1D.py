import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import grad
import matplotlib
matplotlib.rcParams['figure.dpi'] = 200

np.random.seed(2023)
torch.manual_seed(2023)
dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

Length = 1.0
x0 = -1
N = 100
test_size = 200

known_left_ux = 0
bc_left_penalty = 1.0

known_right_tx = 0
bc_right_penalty = 1.0

config = {
    'lr': 0.1,
    'epochs': 30
}

times = {}

def exact(x):
    return 1./135*(68 + 105*x - 40*x**3 + 3*x**5)

def energy(u, x):
    eps = grad(u, x, torch.ones(x.shape, device=dev), create_graph=True, retain_graph=True)[0]
    ### comment why we had to set values < -1 to -1
    ### NaN for (eps + 1)**1.5
    eps[eps < -1] = -1
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

def train_model(data, model, lr=0.5, max_it=20, epochs=10):
    x = torch.from_numpy(data).float()
    # x = x.to(dev)
    x.requires_grad_(True)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=max_it)
    loss = []
    start_time = time.time()
    for i in range(epochs):
        def closure():
            u_pred = (x + 1) * model(x)
            boundary = u_pred*x
            energyInt = energy(u_pred, x)
            energy_loss = (Length - x0)*penalty(energyInt) - (Length - x0)*penalty(boundary)
            # energy_loss = (Length - x0)*energy_pen(energy) - (Length - x0)*energy_pen(boundary)
            if np.isnan(energy_loss.detach().numpy()):
                # eps = grad(u_pred, x, torch.ones(x.shape))[0]
                # plt.subplot(121)
                # plt.plot(x.detach().numpy(), u_pred.detach().numpy())
                # plt.subplot(122)
                # plt.plot(x.detach().numpy(), eps.detach().numpy())
                # plt.show(); exit()
                # print(len(data))
                pass
            else:
                loss = energy_loss
                optimizer.zero_grad()
                loss.backward()
            # print(f'Iter: {i+1:d}, Loss: {energy_loss.item():.5f}')
            return loss
        optimizer.step(closure)


    elapsed = time.time() - start_time
    return model, elapsed

def penalty(input):
    return torch.sum(input) / input.data.nelement()

def energy_pen(input):
    return torch.mean(input)

def L2norm(input, target, x=None, dx=1):
    # print(input.shape, target.shape)
    # print(np.trapz(target**2, dx=dx, axis=0)); exit()
    L2 =  np.sqrt(np.trapz((input-target)**2, dx=dx, axis=0)) / np.sqrt(np.trapz(target**2, dx=dx, axis=0))
    return L2

def evaluate_model(model, test_data):
    x = torch.from_numpy(test_data).float().to(dev)
    x.requires_grad_(True)
    u_pred = (x + 1) * model(x)
    # print(u_pred.shape)
    # print(x.shape)
    # exit()
    energyInt = energy(u_pred, x)
    e_loss = (Length-x0)*(penalty(energyInt) - penalty(u_pred*x))
    # e_loss = (Length-x0)*(energy_pen(energy) - energy_pen(u_pred*x))
    print(f'evaluation_loss: {e_loss}')
    u_pred = u_pred.detach().numpy()
    e_loss = e_loss.detach().numpy()
    return u_pred, e_loss


def plot_Ns():
    model_blank = NN(1, 10, 1)
    test_set = np.array([np.linspace(x0, Length, test_size)]).T
    dxt = test_set[1] - test_set[0]
    times = {}
    losses = {}
    predictions = {}
    ex = exact(test_set)
    # for N in [621]:
    best_loss = np.inf
    Ns = np.logspace(2, 5, 4, dtype=int)
    # for N in np.linspace(100, 1000, 50, dtype=int):
    for N in Ns:
        domain = np.array([np.linspace(x0, Length, N)]).T

        model_trained, elapsed = train_model(domain, model_blank, epochs=config['epochs'], lr=0.1)
        times[N] = elapsed
        # predict
        u_pred, loss = evaluate_model(model_trained, test_set)
        predictions[N] = u_pred
        losses[N] = loss
        loss = np.sum(L2norm(u_pred, ex, dxt))
        if loss < best_loss:
            best_loss = loss
            best_pred = u_pred

    # l1 = sorted(times.items())
    # a,b = zip(*l1)
    # plt.figure(figsize=(4,4))
    # plt.plot(a, b, 'x')
    # plt.show(); exit()

    # l1 = sorted(losses.items())
    # a,b = zip(*l1)
    # plt.figure()
    # plt.plot(a, b)

    markers = {100: 's', 1000: 'o', 10000: 'v', 100000: 'x'}
    lines = {100: None, 1000: None, 10000: '.', 100000: ':'}
    colors = {100: 'tab:blue', 1000: 'tab:orange', 10000: 'tab:red', 100000: 'tab:green'}
    alpha = {100: 0.5, 1000: 0.5, 10000: 0.5, 100000: 1}

    fig1, ax1 = plt.subplots(figsize=(5,4))
    fig2, ax2 = plt.subplots(figsize=(5,4))
    ax1.plot(test_set, exact(test_set))
    for N in Ns:
        xs = N % 97
        print(xs)
        ax1.scatter(test_set[xs::11], predictions[N][xs::11], 
                    c=colors[N], marker=markers[N], s=10, alpha=alpha[N], label=f'N = {N}')
        
        rel_err = (predictions[N][1:]-ex[1:])/ex[1:]
        ax2.semilogy(test_set[1:], rel_err, label=f'N = {N}')
    ax2.legend()
    plt.show()
    # print(test_set.shape, u_pred.shape)


if __name__ == '__main__':
    x = np.linspace(-1, 1, 20_001)
    dx = x[1] - x[0]
    f = x
    y = exact(x)
    # dy = np.gradient(y, dx)
    dy = 1./135*(105 - 120*x**2 + 15*x**4)

    # energy_exact = (1+dy)**(3/2) - 3/2*dy - 1

    # print('Analytisk løsning: ', np.trapz(energy_exact - f*y, x, dx=dx))
    # exit()
    # print(y[:100])
    # plt.plot(x, y, 'o')
    # plt.s<how()
    # plt.plot(x, energy)
    # plt.show(); exit()

    plot_Ns()