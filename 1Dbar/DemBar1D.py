import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import grad
from pathlib import Path
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

def du_exact(x):
    return 1./135*(105 - 120*x**2 + 15*x**4)

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


def save_trained_model(Ns):
    times = {}
    losses = {}
    best_loss = np.inf

    model = NN(1, 10, 1)

    # Ns = np.logspace(2, 5, 4, dtype=int)
    Ns = [100, 500, 1000, 10000]

    for N in Ns:
        domain = np.linspace(x0, Length, N, endpoint=True).reshape((N, 1))

        model, elapsed = train_model(domain, model, epochs=config['epochs'], lr=0.1)
        times[N] = elapsed
        # predict
        u_pred, loss = evaluate_model(model, test_set)
        du_pred = np.gradient(u_pred, (Length-x0)/test_size, axis=0)

        np.save(arrays_path / f'u_pred_{N}', u_pred)
        np.save(arrays_path / f'du_pred_{N}', du_pred)

        # losses[N] = loss
        # loss = np.sum(L2norm(u_pred, exact(test_set), dx))
        # if loss < best_loss:
        #     best_loss = loss
        #     best_pred = u_pred


def plot_Ns(Ns, *filenames):

    predictions = {}
    epsilon_predictions = {}

    for N in Ns:
        predictions[N] = np.load(arrays_path / f'u_pred_{N}.npy')
        epsilon_predictions[N] = np.load(arrays_path / f'du_pred_{N}.npy')

    ex = exact(test_set)
    du_ex = du_exact(test_set)
    # l1 = sorted(times.items())
    # a,b = zip(*l1)
    # plt.figure(figsize=(4,4))
    # plt.plot(a, b, 'x')
    # plt.show(); exit()

    # l1 = sorted(losses.items())
    # a,b = zip(*l1)
    # plt.figure()
    # plt.plot(a, b)

    markers = ['s', 'o', 'v', 'x']
    # lines = [None, None, '.', ':']
    colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green']
    alpha = [0.5, 0.5, 0.5, 1]

    # plot u and relative error
    fig1, ax1 = plt.subplots(figsize=(5,4))
    fig2, ax2 = plt.subplots(figsize=(5,4))

    ax1.plot(test_set, ex, linestyle='-.', color='k', alpha=0.8, label='Exact')
    for i in range(len(Ns)):
        xs = Ns[i] % 97     # different starting point for clearer plot
        ax1.scatter(test_set[xs::11], predictions[Ns[i]][xs::11], 
                    c=colors[i], marker=markers[i], s=10, alpha=alpha[i], label=f'N = {Ns[i]}')
        
        # calculate and plot the relative error
        rel_err = (predictions[Ns[i]][1:] - ex[1:]) #/ ex[1:]      # relative error
        ax2.semilogy(test_set[1:], rel_err, label=f'N = {Ns[i]}')
    ax1.legend()
    ax2.legend()
    # plt.show()

    # plot du and relative error
    fig3, ax3 = plt.subplots(figsize=(5,4))
    fig4, ax4 = plt.subplots(figsize=(5,4))

    ax3.plot(test_set, du_ex, linestyle='-.', color='k', alpha=0.8, label='Exact')
    for i in range(len(Ns)):
        xs = Ns[i] % 97     # different starting point for clearer plot
        ax3.scatter(test_set[xs::11], epsilon_predictions[Ns[i]][xs::11], 
                    c=colors[i], marker=markers[i], s=10, alpha=alpha[i], label=f'N = {Ns[i]}')
        
        # calculate and plot the relative error
        rel_err = (epsilon_predictions[Ns[i]][1:-1] - du_ex[1:-1]) #/ du_ex[1:-1]      # relative error
        ax4.semilogy(test_set[1:-1], rel_err, label=f'N = {Ns[i]}')
    ax3.legend()
    ax4.legend()
    if filenames:
        # if len(filenames != 4):
        #     raise Exception('Need four filenames to save four figures!')
        fig1.savefig(figures_path / Path(filenames[0] + '.pdf'))
        fig2.savefig(figures_path / Path(filenames[1] + '.pdf'))
        fig3.savefig(figures_path / Path(filenames[2] + '.pdf'))
        fig4.savefig(figures_path / Path(filenames[3] + '.pdf'))
    else:
        plt.show()

if __name__ == '__main__':
    test_set = np.linspace(x0, Length, test_size, endpoint=True).reshape((test_size, 1))
    current_path = Path.cwd().resolve()
    figures_path = current_path / 'figures'
    arrays_path = current_path / 'stored_arrays'

    Ns = [100, 500, 1000, 10000]
    save_trained_model(Ns)
    # plot_Ns(Ns, 'fig1', 'fig2', 'fig3', 'fig4')
    plot_Ns(Ns)