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
# sns.set_theme()

np.random.seed(2023)
torch.manual_seed(2023)
# dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dev = torch.device('cpu')

current_path = Path.cwd().resolve()
figures_path = current_path / 'figures'
arrays_path = current_path / 'stored_arrays'
models_path = current_path / 'trained_models'

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
    x.requires_grad_(True)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=max_it)
    total_loss = []
    # start_time = time.time()
    for i in range(epochs):
        # optimizer.zero_grad()
        def closure():
            u_pred = (x + 1) * model(x)
            boundary = u_pred*x
            energyInt = energy(u_pred, x)
            energy_loss = (L - x0)*penalty(energyInt) - (L - x0)*penalty(boundary)
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
                total_loss.append(loss.detach().cpu().numpy())
                # print(f'{id(model):16d}, {len(x):10d}, {loss.item():10.6f}')
                # print('epoch: ', i, 'loss: ', loss)
                optimizer.zero_grad()
                loss.backward()
            # print(f'Iter: {i+1:d}, Loss: {energy_loss.item():.5f}')
            return loss
        optimizer.step(closure)

    # elapsed = time.time() - start_time
    return model, total_loss

def evaluate_model(model, test_data):
    x = torch.from_numpy(test_data).float().to(dev)
    x.requires_grad_(True)
    u_pred = (x + 1) * model(x)
    energyInt = energy(u_pred, x)
    e_loss = (L - x0)*(penalty(energyInt) - penalty(u_pred*x))
    u_pred = u_pred.detach().cpu().numpy()
    e_loss = e_loss.detach().cpu().numpy()
    return u_pred, e_loss


def penalty(input):
    return torch.sum(input) / input.data.nelement()

def energy_pen(input):
    return torch.mean(input)

def L2norm(input, target, x=None, dx=1):
    L2 =  np.sqrt(np.trapz((input-target)**2, dx=dx, axis=0)) / np.sqrt(np.trapz(target**2, dx=dx, axis=0))
    return L2

def train_and_evaluate_model(Ns=1000, lrs=0.1, num_neurons=10, num_epochs=30):
    if isinstance(Ns, (list, tuple)):
        u_norms = np.zeros(len(Ns))
        du_norms = np.zeros(len(Ns))
        for i, N in enumerate(Ns):
            # define model and domain
            model = NN(1, num_neurons, 1)
            domain = np.linspace(x0, L, N, endpoint=True).reshape((N, 1))
            # train mode
            model, total_loss = train_model(domain, model, epochs=num_epochs, lr=lrs)
            # predict u and du
            u_pred, loss = evaluate_model(model, test_set)
            du_pred = np.gradient(u_pred, dx_t, axis=0)
            # calculate L2norm
            u_norms[i] = L2norm(u_pred, u_exact, dx=dx_t)[0]
            du_norms[i] = L2norm(du_pred, du_exact, dx=dx_t)[0]
            
    elif isinstance((lrs and num_neurons), (list, tuple)):
        u_norms = np.zeros((len(lrs), len(num_neurons)))
        du_norms = np.zeros((len(lrs), len(num_neurons)))
        for j, n in enumerate(num_neurons):
            for i, lr in enumerate(lrs):
                # define model and domain
                model = NN(1, n, 1)
                domain = np.linspace(x0, L, Ns, endpoint=True).reshape((Ns, 1))
                # train model
                model, total_loss = train_model(domain, model, epochs=num_epochs, lr=lr)
                # predict u and du
                u_pred, loss = evaluate_model(model, test_set)
                du_pred = np.gradient(u_pred, dx_t, axis=0)
                # calculate L2norm
                u_norms[i, j] = L2norm(u_pred, u_exact, dx=dx_t)[0]
                du_norms[i, j] = L2norm(du_pred, du_exact, dx=dx_t)[0]
    else:
        raise Exception('You need to provide a list of N values or lr AND num_neurons values.')

    return u_norms, du_norms

def calculate_L2norms(Ns=None, lrs=None, num_neurons=None):
    """ Calculate L2 norm for many values of N or lr and num_neurons. """
    predictions = {}
    gradient_predictions = {}

    if Ns:
        u_norms = np.zeros(len(Ns))
        du_norms = np.zeros(len(Ns))
        for i in range(len(Ns)):
            predictions[Ns[i]] = np.load(arrays_path / f'u_pred_N{Ns[i]}.npy')
            gradient_predictions[Ns[i]] = np.load(arrays_path / f'du_pred_N{Ns[i]}.npy')

            u_norms[i] = L2norm(predictions[Ns[i]], ex, dx=dx_t)[0]
            du_norms[i] = L2norm(gradient_predictions[Ns[i]], du_exact, dx=dx_t)[0]

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
                du_norms[i, j] = L2norm(gradient_predictions[(lr, n)], du_exact, dx=dx_t)[0]

    else:
        raise Exception('sYou need to provide a list of N values or lr AND num_neurons values.')
    
    return u_norms, du_norms

def save_trained_model_N(Ns, lr=0.05, num_neurons=5, num_epochs=30):
    for N in Ns:
        model = NN(1, num_neurons, 1)
        domain = np.linspace(x0, L, N, endpoint=True).reshape((N, 1))

        model, total_loss = train_model(domain, model, epochs=num_epochs, lr=lr)
        u_pred, loss = evaluate_model(model, test_set)
        du_pred = np.gradient(u_pred, dx_t, axis=0)

        # np.save(arrays_path / f'u_pred_N{N}', u_pred)
        # np.save(arrays_path / f'du_pred_N{N}', du_pred)
        # np.save(arrays_path / f'total_loss{N}', np.array(total_loss))

def save_trained_model_lr_neurons(lrs, num_neurons, N=1000, num_epochs=30):
    for n in num_neurons:
        for lr in lrs:
            model = NN(1, n, 1)
            
            domain = np.linspace(x0, L, N, endpoint=True).reshape((N, 1))
            model, total_loss = train_model(domain, model, epochs=num_epochs, lr=lr)
            u_pred, loss = evaluate_model(model, test_set)
            du_pred = np.gradient(u_pred, dx_t, axis=0)

            np.save(arrays_path / f'u_pred_lr{lr}_n{n}', u_pred)
            np.save(arrays_path / f'du_pred_lr{lr}_n{n}', du_pred)

def plot_Ns(Ns, *filenames):
    predictions = {}
    gradient_predictions = {}

    for N in Ns:
        predictions[N] = np.load(arrays_path / f'u_pred_N{N}.npy')
        gradient_predictions[N] = np.load(arrays_path / f'du_pred_N{N}.npy')

    u_exact = exact(test_set)
    du_exact = du_exact(test_set)

    markers = ['s', 'o', 'v', 'x']
    colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green']
    alpha = [0.5, 0.5, 0.5, 1]

    # plot u and absative error
    fig1, ax1 = plt.subplots(figsize=(5,4))
    fig2, ax2 = plt.subplots(figsize=(5,4))

    ax1.plot(test_set, u_exact, linestyle='-.', color='k', alpha=0.8, label='Exact')
    for i in range(len(Ns)):
        xs = 0
        ax1.scatter(test_set[xs::11], predictions[Ns[i]][xs::11], 
                    c=colors[i], marker=markers[i], s=10, alpha=alpha[i], label=f'N = {Ns[i]}')
        
        # calculate and plot the absolute error
        abs_err = (predictions[Ns[i]] - u_exact)
        ax2.semilogy(test_set, abs_err, label=f'N = {Ns[i]}')

    ax1.set_title('Displacement as function of length')
    ax2.set_title('Absolute error of displacement')
    ax1.set_xlabel('X')
    ax2.set_xlabel('X')
    ax1.set_ylabel('Displacement')
    ax2.set_ylabel('Absolute error')
    ax1.grid()
    ax2.grid()
    ax1.legend()
    ax2.legend()
    # plt.show()

    # plot du and relative error
    fig3, ax3 = plt.subplots(figsize=(5,4))
    fig4, ax4 = plt.subplots(figsize=(5,4))

    ax3.plot(test_set, du_exact, linestyle='-.', color='k', alpha=0.8, label='Exact')
    for i in range(len(Ns)):
        xs = 0     # different starting point for clearer plot
        ax3.scatter(test_set[xs::11], gradient_predictions[Ns[i]][xs::11], 
                    c=colors[i], marker=markers[i], s=10, alpha=alpha[i], label=f'N = {Ns[i]}')
        
        # calculate and plot the relative error
        abs_err = (gradient_predictions[Ns[i]] - du_exact) #     # absolute error
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
        fig1.savefig(figures_path / Path('fig0.pdf'))
        fig2.savefig(figures_path / Path('fig1.pdf'))
        fig3.savefig(figures_path / Path('fig2.pdf'))
        fig4.savefig(figures_path / Path('fig3.pdf'))


if __name__ == '__main__':
    test_set = np.linspace(x0, L, test_size+2, endpoint=True)[1:-1].reshape((test_size, 1))
    u_exact = exact(test_set); du_exact = dexact(test_set)

    num_expreriments = 30

    ### Train and plot for learning rate and number of neurons
    ### make own function?
    # lrs = [0.01, 0.05, .1, .5, 1]
    # num_neurons = [5, 10, 15, 20, 30]
    # u_norms = np.zeros((len(lrs), len(num_neurons)))
    # du_norms = u_norms.copy()
    # for _ in range(num_expreriments):
    #     u_norms_i, du_norms_i = train_and_evaluate_model(Ns=1000, lrs=lrs, num_neurons=num_neurons)
    #     u_norms += u_norms_i
    #     du_norms += du_norms_i

    # u_norms /= num_expreriments
    # du_norms /= num_expreriments

    # y_ticks=[str(i) for i in lrs]
    # x_ticks=[str(i) for i in num_neurons]
    # fig1, ax1 = plt.subplots(figsize=(5,5))
    # fig2, ax2 = plt.subplots(figsize=(5,5))
    # sns.heatmap(u_norms, annot=True, ax=ax1, 
    #             cmap='cividis', xticklabels=x_ticks,
    #             yticklabels=y_ticks, cbar=False, vmax=np.max(u_norms[u_norms < 1]))
    # sns.heatmap(du_norms, annot=True, ax=ax2, 
    #             cmap='cividis', xticklabels=x_ticks,
    #             yticklabels=y_ticks, cbar=False, vmax=np.max(du_norms[du_norms < 1]))
    # ax1.set_title(r'$L^2$ error norm for displacement')
    # ax2.set_title(r'$L^2$ error norm for displacement gradient')
    # ax1.set_xlabel('Nr. of neurons in hidden layer')
    # ax2.set_xlabel('Nr. of neurons in hidden layer')
    # ax1.set_ylabel(r'$\eta$')
    # ax2.set_ylabel(r'$\eta$')
    # fig1.savefig(figures_path / 'bar_heatmap_lr_neurons1.pdf')
    # fig2.savefig(figures_path / 'bar_heatmap_lr_neurons2.pdf')

    Ns = [100, 500, 1000, 10000]
    # lrs = [.1, .1, .5]
    # num_neurons = [5, 10, 5]
    # u_norms = np.zeros((len(Ns), len(lrs)))
    # du_norms = u_norms.copy()
    # for i, (lr, n) in enumerate(zip(lrs, num_neurons)):
    #     for _ in range(num_expreriments):
    #         u_norms_i, du_norms_i = train_and_evaluate_model(Ns, lrs=lr, num_neurons=n)
    #         u_norms[:, i] += u_norms_i
    #         du_norms[:, i] += du_norms_i
    
    # u_norms /= num_expreriments
    # du_norms /= num_expreriments

    # y_ticks=[str(i) for i in Ns]
    # x_ticks=[f'({i}, {j})' for i, j in zip(lrs, num_neurons)]
    # fig1, ax1 = plt.subplots(figsize=(5,5))
    # fig2, ax2 = plt.subplots(figsize=(5,5))
    # sns.heatmap(u_norms, annot=True, ax=ax1, 
    #             cmap='cividis', xticklabels=x_ticks,
    #             yticklabels=y_ticks, cbar=False, vmax=np.max(u_norms[u_norms < 1]))
    #             # yticklabels=y_ticks, cbar=False, vmax=np.median(u_norms))
    # sns.heatmap(du_norms, annot=True, ax=ax2, 
    #             cmap='cividis', xticklabels=x_ticks,
    #             yticklabels=y_ticks, cbar=False, vmax=np.max(du_norms[du_norms < 1]))
    #             # yticklabels=y_ticks, cbar=False, vmax=np.median(du_norms))
    # ax1.set_title(r'$L^2$ error norm for displacement')
    # ax2.set_title(r'$L^2$ error norm for displacement gradient')
    # ax1.set_xlabel(r'($\eta$, Nr. of neurons in hidden layer)')
    # ax2.set_xlabel(r'($\eta$, Nr. of neurons in hidden layer)')
    # ax1.set_ylabel(r'$N$')
    # ax2.set_ylabel(r'$N$')
    # fig1.savefig(figures_path / 'bar_heatmap_Ns1.pdf')
    # fig2.savefig(figures_path / 'bar_heatmap_Ns2.pdf')



    # train_and_evaluate_model(Ns, num_epochs=3)
    # plot_Ns(Ns, 'dem_fig1', 'dem_fig2', 'dem_fig3', 'dem_fig4')
    # for N in Ns:
    #     total_loss = np.load(arrays_path / f'total_loss{N}.npy')
    #     plt.figure()
    #     plt.plot(total_loss)
    #     plt.savefig(figures_path / f'loss_{N}.pdf')