import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import grad
from pathlib import Path
import matplotlib
matplotlib.rcParams['figure.dpi'] = 200
import seaborn as sns
sns.set()
from scipy.integrate import simpson as simpson_scipy
import sys
sys.path.insert(0, "..")
from simps import simpson


np.random.seed(2023)
torch.manual_seed(2023)
# dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dev = torch.device('cpu')

current_path = Path.cwd().resolve()
figures_path = current_path / 'figures'
arrays_path = current_path / 'stored_arrays'

L = 1.0
x0 = -1
N = 100
test_size = 200
dx_t = (L - x0) / test_size
# print(dx_t)
known_left_ux = 0
bc_left_penalty = 1.0

known_right_tx = 0
bc_right_penalty = 1.0

times = {}

def exact(x):
    return 1./135*(68 + 105*x - 40*x**3 + 3*x**5)

def dexact(x):
    return 1./135*(105 - 120*x**2 + 15*x**4)

def energy(u, x):
    eps = grad(u, x, torch.ones(x.shape, device=dev), 
               create_graph=True, retain_graph=True)[0]
    # fix NaN for (eps + 1)**1.5
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

def train_model(data, model, lr=0.1, epochs=10):
    x = torch.from_numpy(data).float().to(dev)
    model.to(dev)
    dx = x[1] - x[0]
    x.requires_grad_(True)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=lr)
    total_loss = torch.zeros((epochs, len(x))).to(dev)
    # start_time = time.time()
    for i in range(epochs):
        # optimizer.zero_grad()
        def closure():
            u_pred = (x + 1) * model(x)
            boundary = u_pred*x
            energyInt = energy(u_pred, x)
            energy_loss = (simpson(energyInt, dx=dx, axis=0) - simpson(boundary, dx=dx, axis=0))
            # energy_loss = (L-x0)*(penalty(energyInt) - penalty(boundary))

            loss = energy_loss
            total_loss[i] = energy_loss

            optimizer.zero_grad()
            loss.backward()
            # print(f'Iter: {i+1:5d}, Loss: {energy_loss.item():8.5f}')
            return loss
        optimizer.step(closure)

    # elapsed = time.time() - start_time
    return model, total_loss

def penalty(x):
    return torch.sum(x) / x.data.nelement()

def evaluate_model(test_data, model):
    x = torch.from_numpy(test_data).float().to(dev)
    x.requires_grad_(True)
    u_pred = (x + 1) * model(x)
    energyInt = energy(u_pred, x)
    e_loss = simpson(energyInt) - simpson(u_pred*x)
    u_pred = u_pred.detach().cpu().numpy()
    e_loss = e_loss.detach().cpu().numpy()
    return u_pred, e_loss

def L2norm(input, target, dx):
    L2 =  np.sqrt(
            simpson_scipy((input - target)**2, dx=dx, axis=0)
          / simpson_scipy(target**2, dx=dx, axis=0))
    return L2

if __name__ == '__main__':
    x_test = np.linspace(x0, L, test_size+2)[1:-1].reshape((test_size, 1))
    dx = x_test[1] - x_test[0]

    u_exact = exact(x_test)
    du_exact = dexact(x_test)

    # Ns = [50, 100, 250, 500, 1000, 2500, 5000]
    Ns = [10, 100, 1000]
    e = np.zeros(len(Ns))

    u_fem = np.load('u_fem.npy')
    du_fem = np.gradient(u_fem, dx[0])

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()

    for i, N in enumerate(Ns):
        model = NN(1, 30, 1)
        x = np.linspace(x0, L, N).reshape((N, 1))
        model, loss = train_model(x, model, epochs=30)
        u_pred, _ = evaluate_model(x_test, model)
        del model
        e[i] += L2norm(u_pred, u_exact, dx[0])[0]

        ax1.semilogy(x_test, np.abs(u_pred - u_exact), label=f'N = {N}')
        du = np.gradient(u_pred[:, 0], dx[0]).reshape((test_size, 1))

        ax4.semilogy(x_test, np.abs(du - du_exact), label=f'N = {N}')

        if N == 100:
            ax2.plot(x_test, u_exact, label='Exact')
            ax2.plot(x_test, u_fem, '--', label='FEM')
            ax2.plot(x_test, u_pred, 'g:', label='DEM')

            ax3.plot(x_test, du_exact, label=f'Exact')
            ax3.plot(x_test, du_fem, '--', label='FEM')
            ax3.plot(x_test, du, 'g:', label=f'DEM')
    
    ax1.set_xlabel('$X$')
    ax1.set_ylabel('Absolute error')
    ax1.legend(loc='upper right')
    fig1.savefig('figures/error_u_dem.pdf')

    ax2.set_xlabel('$X$')
    ax2.set_ylabel('$u$')
    ax2.legend(loc='lower right')
    fig2.savefig('figures/u.pdf')

    ax3.set_xlabel('$X$')
    ax3.set_ylabel('du/dX')
    ax3.legend(loc='upper right')
    fig3.savefig('figures/du.pdf')

    ax4.set_xlabel('$X$')
    ax4.set_ylabel('Absolute error')
    ax4.legend(loc='upper right')
    fig4.savefig('figures/error_du_dem.pdf')
    plt.show()
