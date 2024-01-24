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

L = 1.0
x0 = -1
N = 100
test_size = 200
dx_t = (L - x0) / test_size
print(dx_t)
known_left_ux = 0
bc_left_penalty = 1.0

known_right_tx = 0
bc_right_penalty = 1.0

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
    x = torch.from_numpy(data).float().to(dev)
    model.to(dev)
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
            energy_loss = (L - x0)*penalty(energyInt) - (L - x0)*penalty(boundary)
            # energy_loss = (L - x0)*energy_pen(energy) - (L - x0)*energy_pen(boundary)
            # if np.isnan(energy_loss.detach().cpu().numpy()):
            if torch.isnan(energy_loss):
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
                print(loss)
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
    L2 =  np.sqrt(np.trapz((input-target)**2, dx=dx, axis=0)) / np.sqrt(np.trapz(target**2, dx=dx, axis=0))
    return L2

def evaluate_model(model, test_data):
    x = torch.from_numpy(test_data).float().to(dev)
    x.requires_grad_(True)
    u_pred = (x + 1) * model(x)
    energyInt = energy(u_pred, x)
    e_loss = (L - x0)*(penalty(energyInt) - penalty(u_pred*x))
    # e_loss = (L-x0)*(energy_pen(energy) - energy_pen(u_pred*x))
    # print(f'evaluation_loss: {e_loss}')
    u_pred = u_pred.detach().cpu().numpy()
    e_loss = e_loss.detach().cpu().numpy()
    return u_pred, e_loss


def save_trained_model_N(Ns, lr=0.1, num_neurons=10, num_epochs=30):
    times = {}
    losses = {}
    best_loss = np.inf

    model = NN(1, num_neurons, 1)

    # Ns = np.logspace(2, 5, 4, dtype=int)
    Ns = [100, 500, 1000, 10000]

    for N in Ns:
        domain = np.linspace(x0, L, N, endpoint=True).reshape((N, 1))

        model, elapsed = train_model(domain, model, epochs=num_epochs, lr=lr)
        times[N] = elapsed
        # predict
        u_pred, loss = evaluate_model(model, test_set)
        du_pred = np.gradient(u_pred, dx_t, axis=0)

        np.save(arrays_path / f'u_pred_N{N}', u_pred)
        np.save(arrays_path / f'du_pred_N{N}', du_pred)

        # losses[N] = loss
        # loss = np.sum(L2norm(u_pred, exact(test_set), dx))
        # if loss < best_loss:
        #     best_loss = loss
        #     best_pred = u_pred

def save_trained_model_lr_neurons(lrs, num_neurons, N=1000, num_epochs=30):
    # times = {}
    # losses = {}
    # best_loss = np.inf
    for n in num_neurons:
        for lr in lrs:
            print('lr: ', lr, 'n: ', n)
            model = NN(1, n, 1)
            domain = np.linspace(x0, L, N, endpoint=True).reshape((N, 1))

            model, elapsed = train_model(domain, model, epochs=num_epochs, lr=lr)
            times[N] = elapsed
            # predict
            u_pred, loss = evaluate_model(model, test_set)
            du_pred = np.gradient(u_pred, dx_t, axis=0)

            np.save(arrays_path / f'u_pred_lr{lr}_n{n}', u_pred)
            np.save(arrays_path / f'du_pred_lr{lr}_n{n}', du_pred)

            # losses[N] = loss
            # loss = np.sum(L2norm(u_pred, exact(test_set), dx))
            # if loss < best_loss:
            #     best_loss = loss
            #     best_pred = u_pred

def plot_Ns(Ns, *filenames):
    predictions = {}
    gradient_predictions = {}

    for N in Ns:
        predictions[N] = np.load(arrays_path / f'u_pred_N{N}.npy')
        gradient_predictions[N] = np.load(arrays_path / f'du_pred_N{N}.npy')

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

    # plot u and absative error
    fig1, ax1 = plt.subplots(figsize=(5,4))
    fig2, ax2 = plt.subplots(figsize=(5,4))

    ax1.plot(test_set, ex, linestyle='-.', color='k', alpha=0.8, label='Exact')
    for i in range(len(Ns)):
        xs = Ns[i] % 97     # different starting point for clearer plot
        ax1.scatter(test_set[xs::11], predictions[Ns[i]][xs::11], 
                    c=colors[i], marker=markers[i], s=10, alpha=alpha[i], label=f'N = {Ns[i]}')
        
        # calculate and plot the absolute error
        abs_err = (predictions[Ns[i]] - ex)
        ax2.semilogy(test_set, abs_err, label=f'N = {Ns[i]}')

    ax1.set_title('Displacement as function of length')
    ax2.set_title('Absolute error of displacement')
    ax1.set_xlabel('X')
    ax2.set_xlabel('X')
    ax1.set_ylabel('Displacement')
    ax2.set_ylabel('Absolute error')
    ax1.legend()
    ax2.legend()
    # plt.show()

    # plot du and relative error
    fig3, ax3 = plt.subplots(figsize=(5,4))
    fig4, ax4 = plt.subplots(figsize=(5,4))

    ax3.plot(test_set, du_ex, linestyle='-.', color='k', alpha=0.8, label='Exact')
    for i in range(len(Ns)):
        xs = Ns[i] % 97     # different starting point for clearer plot
        ax3.scatter(test_set[xs::11], gradient_predictions[Ns[i]][xs::11], 
                    c=colors[i], marker=markers[i], s=10, alpha=alpha[i], label=f'N = {Ns[i]}')
        
        # calculate and plot the relative error
        abs_err = (gradient_predictions[Ns[i]] - du_ex) #    / du_ex      # absative error
        ax4.semilogy(test_set, abs_err, label=f'N = {Ns[i]}')

    ax3.set_title('Displacement gradient as function of length')
    ax4.set_title('Absolute error of displacement gradient')
    ax3.set_xlabel('X')
    ax4.set_xlabel('X')
    ax3.set_ylabel('Displacement gradient')
    ax4.set_ylabel('Relative error')
    ax3.legend()
    ax4.legend()

    if filenames:
        if len(filenames) != 4:
            raise Exception('Need four filenames to save four figures!')
        fig1.savefig(figures_path / Path(filenames[0] + '.pdf'))
        fig2.savefig(figures_path / Path(filenames[1] + '.pdf'))
        fig3.savefig(figures_path / Path(filenames[2] + '.pdf'))
        fig4.savefig(figures_path / Path(filenames[3] + '.pdf'))
    else:
        plt.show()

def calculate_L2norms(Ns=None, lrs=None, num_neurons=None):
    """ Calculate L2 norm for many values of N or lr and num_neurons. """
    predictions = {}
    gradient_predictions = {}
    # u_norms = {}
    # du_norms = {}

    if Ns:
        u_norms = np.zeros(len(Ns))
        du_norms = np.zeros(len(Ns))
        for i in range(len(Ns)):
            predictions[Ns[i]] = np.load(arrays_path / f'u_pred_N{Ns[i]}.npy')
            gradient_predictions[Ns[i]] = np.load(arrays_path / f'du_pred_N{Ns[i]}.npy')

            u_norms[i] = L2norm(predictions[Ns[i]], ex, dx=dx_t)[0]
            du_norms[i] = L2norm(gradient_predictions[Ns[i]], du_ex, dx=dx_t)[0]
            
    elif (lrs and num_neurons):
        u_norms = np.zeros((len(lrs), len(num_neurons)))
        du_norms = np.zeros((len(lrs), len(num_neurons)))
        for i in range(len(lrs)):
            lr = lrs[i]
            for j in range(len(num_neurons)):
                n = num_neurons[j]
                predictions[(lr, n)] = np.load(arrays_path / f'u_pred_lr{lr}_n{n}.npy')
                gradient_predictions[(lr, n)] = np.load(arrays_path / f'du_pred_lr{lr}_n{n}.npy')

                u_norms[i, j] = L2norm(predictions[(lr, n)], ex, dx=dx_t)[0]
                du_norms[i, j] = L2norm(gradient_predictions[(lr, n)], du_ex, dx=dx_t)[0]

    else:
        print('Feil!')
        return
    return u_norms, du_norms

if __name__ == '__main__':
    test_set = np.linspace(x0, L, test_size, endpoint=True).reshape((test_size, 1))
    ex = exact(test_set); du_ex = du_exact(test_set)

    current_path = Path.cwd().resolve()
    figures_path = current_path / 'figures'
    arrays_path = current_path / 'stored_arrays'

    Ns = [100, 500, 1000, 10000]
    # save_trained_model_N(Ns)
    # plot_Ns(Ns, 'dem_fig1', 'dem_fig2', 'dem_fig3', 'dem_fig4')
    # plot_Ns(1000*Ns)
    num_expreriments = 50

    # u_norms = np.zeros(len(Ns))
    # du_norms = u_norms.copy()

    # for _ in range(num_expreriments):
    #     save_trained_model_lr_neurons([.1, .2, .3], [10, 20, 30])
        # save_trained_model_N(Ns)
        # u_norms += calculate_L2norms(Ns=Ns)[0]
        # du_norms += calculate_L2norms(Ns=Ns)[1]

    # print(u_norms/num_expreriments)
    # print(du_norms/num_expreriments)
        
    lrs = [0.01, 0.05, .1, .5, 1]
    num_neurons = [5, 10, 15, 20, 30]
    u_norms = np.zeros((len(lrs), len(num_neurons)))
    du_norms = u_norms.copy()
    for _ in range(num_expreriments):
        save_trained_model_lr_neurons(lrs, num_neurons)
        u_norms += calculate_L2norms(lrs=lrs, num_neurons=num_neurons)[0]
        du_norms += calculate_L2norms(lrs=lrs, num_neurons=num_neurons)[1]

    # print(u_norms/num_expreriments)
    # print(du_norms/num_expreriments)
    u_norms /= num_expreriments
    du_norms /= num_expreriments

    import seaborn as sns
    sns.set()
    y_ticks=[str(i) for i in lrs]
    x_ticks=[str(i) for i in num_neurons]
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    sns.heatmap(u_norms, annot=True, ax=ax[0], cmap='cividis', xticklabels=x_ticks, yticklabels=y_ticks, cbar=False)
    sns.heatmap(du_norms, annot=True, ax=ax[1], cmap='cividis', xticklabels=x_ticks, yticklabels=y_ticks, cbar=False)
    ax[0].set_xlabel('Nr. of neurons in hidden layer')
    ax[1].set_xlabel('Nr. of neurons in hidden layer')
    ax[0].set_ylabel('$\eta$')
    plt.savefig(figures_path / 'heatmap_lr_neurons.pdf')
    # plt.show()

