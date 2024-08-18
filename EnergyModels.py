import torch
from torch.autograd import grad
from DEM import dev

class NeoHookeanEnergyModel:
    def __init__(self, lmbda, mu):
        self.lmbda = lmbda
        self.mu = mu

    def noe(self, u, x):
        duxdxyz = grad(u[:, 0].unsqueeze(1), x, 
                        torch.ones(x.shape[0], 1, device=dev), 
                        create_graph=True, retain_graph=True)[0]
        duydxyz = grad(u[:, 1].unsqueeze(1), x, 
                        torch.ones(x.shape[0], 1, device=dev), 
                        create_graph=True, retain_graph=True)[0]
        duzdxyz = grad(u[:, 2].unsqueeze(1), x, 
                        torch.ones(x.shape[0], 1, device=dev), 
                        create_graph=True, retain_graph=True)[0]

        Fxx = duxdxyz[:, 0].unsqueeze(1) + 1
        Fxy = duxdxyz[:, 1].unsqueeze(1) + 0
        Fxz = duxdxyz[:, 2].unsqueeze(1) + 0
        Fyx = duydxyz[:, 0].unsqueeze(1) + 0
        Fyy = duydxyz[:, 1].unsqueeze(1) + 1
        Fyz = duydxyz[:, 2].unsqueeze(1) + 0
        Fzx = duzdxyz[:, 0].unsqueeze(1) + 0
        Fzy = duzdxyz[:, 1].unsqueeze(1) + 0
        Fzz = duzdxyz[:, 2].unsqueeze(1) + 1

        self.Fxx = Fxx; self.Fxy = Fxy; self.Fxz = Fxz
        self.Fyx = Fyx; self.Fyy = Fyy; self.Fyz = Fyz
        self.Fzx = Fzx; self.Fzy = Fzy; self.Fzz = Fzz

        detF = (Fxx * (Fyy * Fzz - Fyz * Fzy) 
                - Fxy * (Fyx * Fzz - Fyz * Fzx) 
                + Fxz * (Fyx * Fzy - Fyy * Fzx))
        trC = (Fxx ** 2 + Fxy ** 2 + Fxz ** 2 
               + Fyx ** 2 + Fyy ** 2 + Fyz ** 2 
               + Fzx ** 2 + Fzy ** 2 + Fzz ** 2)

        return detF, trC
    
    def get_passive_strain_energy(self, trC):
        return 0.5 * self.mu * (trC - 3)
    
    def get_active_strain_energy(self, u, x):
        raise NotImplementedError('Active strain energy function is not implemented for this case!')
    
    def get_compressibility(self, detF):
        return 0.5 * self.lmbda * (torch.log(detF) * torch.log(detF)) - self.mu * torch.log(detF)
        
    def __call__(self, u, x, J=False):
        ### energy frunction from DEM paper ### 
        # duxdxyz = grad(u[:, 0].unsqueeze(1), x, torch.ones(x.shape[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        # duydxyz = grad(u[:, 1].unsqueeze(1), x, torch.ones(x.shape[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        # duzdxyz = grad(u[:, 2].unsqueeze(1), x, torch.ones(x.shape[0], 1, device=dev), create_graph=True, retain_graph=True)[0]

        # Fxx = duxdxyz[:, 0].unsqueeze(1) + 1
        # Fxy = duxdxyz[:, 1].unsqueeze(1) + 0
        # Fxz = duxdxyz[:, 2].unsqueeze(1) + 0
        # Fyx = duydxyz[:, 0].unsqueeze(1) + 0
        # Fyy = duydxyz[:, 1].unsqueeze(1) + 1
        # Fyz = duydxyz[:, 2].unsqueeze(1) + 0
        # Fzx = duzdxyz[:, 0].unsqueeze(1) + 0
        # Fzy = duzdxyz[:, 1].unsqueeze(1) + 0
        # Fzz = duzdxyz[:, 2].unsqueeze(1) + 1

        # detF = Fxx * (Fyy * Fzz - Fyz * Fzy) - Fxy * (Fyx * Fzz - Fyz * Fzx) + Fxz * (Fyx * Fzy - Fyy * Fzx)
        # trC = Fxx ** 2 + Fxy ** 2 + Fxz ** 2 + Fyx ** 2 + Fyy ** 2 + Fyz ** 2 + Fzx ** 2 + Fzy ** 2 + Fzz ** 2
        detF, trC = self.noe(u, x)
        
        StrainEnergy = (self.get_compressibility(detF) 
                        + self.get_passive_strain_energy(trC))

        if J:
            return StrainEnergy, detF
        return StrainEnergy
    
class NeoHookeanActiveEnergyModel(NeoHookeanEnergyModel):
    def __init__(self, mu, Ta=1.0, f0=torch.tensor([1, 0, 0]), kappa=1E3):
        super().__init__(0.0, mu)
        self.Ta = Ta
        self.f0 = f0
        self.kappa = kappa

    def get_active_strain_energy(self, detF, I4f0):
        return 0.5 * self.Ta / detF * (I4f0 - 1)

    def get_compressibility(self, detF):
        return 0.5 * self.kappa * (detF - 1)**2

    def noe(self, u, x):
        f0 = self.f0
        # duxdxyz = grad(u[:, 0].unsqueeze(1), x, torch.ones(x.shape[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        # duydxyz = grad(u[:, 1].unsqueeze(1), x, torch.ones(x.shape[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        # duzdxyz = grad(u[:, 2].unsqueeze(1), x, torch.ones(x.shape[0], 1, device=dev), create_graph=True, retain_graph=True)[0]

        # Fxx = duxdxyz[:, 0].unsqueeze(1) + 1
        # Fxy = duxdxyz[:, 1].unsqueeze(1) + 0
        # Fxz = duxdxyz[:, 2].unsqueeze(1) + 0
        # Fyx = duydxyz[:, 0].unsqueeze(1) + 0
        # Fyy = duydxyz[:, 1].unsqueeze(1) + 1
        # Fyz = duydxyz[:, 2].unsqueeze(1) + 0
        # Fzx = duzdxyz[:, 0].unsqueeze(1) + 0
        # Fzy = duzdxyz[:, 1].unsqueeze(1) + 0
        # Fzz = duzdxyz[:, 2].unsqueeze(1) + 1
        detF, trC = super().noe(u, x)

        Fxx = self.Fxx; Fxy = self.Fxy; Fxz = self.Fxz
        Fyx = self.Fyx; Fyy = self.Fyy; Fyz = self.Fyz
        Fzx = self.Fzx; Fzy = self.Fzy; Fzz = self.Fzz

        C11 = Fxx*Fxx + Fyx*Fyx + Fzx*Fzx
        C12 = Fxx*Fxy + Fyx*Fyy + Fzx*Fzy
        C13 = Fxx*Fxz + Fyx*Fyz + Fzx*Fzz
        C21 = Fxy*Fxx + Fyy*Fyx + Fzy*Fzx
        C22 = Fxy*Fxy + Fyy*Fyy + Fzy*Fzy
        C23 = Fxy*Fxz + Fyy*Fyz + Fzy*Fzz
        C31 = Fxz*Fxx + Fyz*Fyx + Fzz*Fzx
        C32 = Fxz*Fxy + Fyz*Fyy + Fzz*Fzy
        C33 = Fxz*Fxz + Fyz*Fyz + Fzz*Fzz

        I4f0 = (f0[0] * (f0[0]*C11 + f0[1]*C21 + f0[2]*C31)
        + f0[1] * (f0[0]*C12 + f0[1]*C22 + f0[2]*C32)
        + f0[2] * (f0[0]*C13 + f0[1]*C23 + f0[2]*C33))

        return detF, trC, I4f0

    def __call__(self, u, x, J=False):
        detF, trC, I4f0 = self.noe(u, x)

        StrainEnergy = (self.get_compressibility(detF)
                        + self.get_passive_strain_energy(trC)
                        + self.get_active_strain_energy(detF, I4f0))

        if J:
            return StrainEnergy, detF
        return StrainEnergy
    
class GuccioneEnergyModel:
    def __init__(self, C, bf, bt, bfs, kappa=1E3):
        self.C = C
        self.bf = bf
        self.bt = bt
        self.bfs = bfs
        self.kappa = kappa

    def get_W(self, Q):
        return self.C / 2 * (torch.exp(Q) - 1)

    def get_compressibility(self, detF):
        return  0.5 * self.kappa * (detF - 1)**2

    def noe(self, u, x):
        # Guccione energy mode. Get source from verification paper!!!
        duxdxyz = grad(u[:, 0].unsqueeze(1), x, 
                        torch.ones(x.shape[0], 1, device=dev), 
                        create_graph=True, retain_graph=True)[0]
        duydxyz = grad(u[:, 1].unsqueeze(1), x, 
                        torch.ones(x.shape[0], 1, device=dev), 
                        create_graph=True, retain_graph=True)[0]
        duzdxyz = grad(u[:, 2].unsqueeze(1), x, 
                        torch.ones(x.shape[0], 1, device=dev), 
                        create_graph=True, retain_graph=True)[0]

        Fxx = duxdxyz[:, 0].unsqueeze(1) + 1
        Fxy = duxdxyz[:, 1].unsqueeze(1) + 0
        Fxz = duxdxyz[:, 2].unsqueeze(1) + 0
        Fyx = duydxyz[:, 0].unsqueeze(1) + 0
        Fyy = duydxyz[:, 1].unsqueeze(1) + 1
        Fyz = duydxyz[:, 2].unsqueeze(1) + 0
        Fzx = duzdxyz[:, 0].unsqueeze(1) + 0
        Fzy = duzdxyz[:, 1].unsqueeze(1) + 0
        Fzz = duzdxyz[:, 2].unsqueeze(1) + 1

        E11 = 0.5*(Fxx*Fxx + Fyx*Fyx + Fzx*Fzx - 1)
        E12 = 0.5*(Fxx*Fxy + Fyx*Fyy + Fzx*Fzy - 0)
        E13 = 0.5*(Fxx*Fxz + Fyx*Fyz + Fzx*Fzz - 0)
        E21 = 0.5*(Fxy*Fxx + Fyy*Fyx + Fzy*Fzx - 0)
        E22 = 0.5*(Fxy*Fxy + Fyy*Fyy + Fzy*Fzy - 1)
        E23 = 0.5*(Fxy*Fxz + Fyy*Fyz + Fzy*Fzz - 0)
        E31 = 0.5*(Fxz*Fxx + Fyz*Fyx + Fzz*Fzx - 0)
        E32 = 0.5*(Fxz*Fxy + Fyz*Fyy + Fzz*Fzy - 0)
        E33 = 0.5*(Fxz*Fxz + Fyz*Fyz + Fzz*Fzz - 1)

        detF = Fxx * (Fyy * Fzz - Fyz * Fzy) - Fxy * (Fyx * Fzz - Fyz * Fzx) + Fxz * (Fyx * Fzy - Fyy * Fzx)
        Q = (self.bf*E11**2
             + self.bt*(E22**2 + E33**2 + E23**2 + E32**2)
             + self.bfs*(E12**2 + E21**2 + E13**2 + E31**2))

        return Q, detF
        # W = self.C / 2 * (torch.exp(Q) - 1)
        # compressibility = kappa/2 * (detF - 1)**2

    def __call__(self, u, x, J=False):
        Q, detF = self.noe(u, x)

        W = self.get_W(Q)
        compressibility = self.get_compressibility(detF)
        total_energy = W + compressibility

        if J:
            return total_energy, detF
        return total_energy