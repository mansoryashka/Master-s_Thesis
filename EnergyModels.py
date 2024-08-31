import torch
from torch.autograd import grad
from DEM import dev

class NeoHookeanEnergyModel:
    def __init__(self, lmbda, mu):
        self.lmbda = lmbda
        self.mu = mu

    def _get_invariants(self, u, x):
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
    
    # def _get_active_strain_energy(self, u, x):
    #     raise NotImplementedError('Active strain energy function is not implemented for this case!')
    
    def _get_passive_strain_energy(self, trC):
        return 0.5 * self.mu * (trC - 3)
    
    def _get_compressibility(self, detF):
        return 0.5 * self.lmbda * (torch.log(detF) * torch.log(detF)) - self.mu * torch.log(detF)
        
    def __call__(self, u, x):
        ### energy frunction from DEM paper ### 
        detF, trC = self._get_invariants(u, x)
        
        StrainEnergy = (self._get_compressibility(detF) 
                      + self._get_passive_strain_energy(trC))

        return StrainEnergy
    
class NeoHookeanActiveEnergyModel(NeoHookeanEnergyModel):
    def __init__(self, mu, Ta=1.0, f0=torch.tensor([1, 0, 0]), kappa=1E3):
        super().__init__(0.0, mu)
        self.Ta = Ta
        self.f0 = f0
        self.kappa = kappa

    def _get_active_strain_energy(self, detF, I4f0):
        return 0.5 * self.Ta / detF * (I4f0 - 1)

    def _get_compressibility(self, detF):
        return 0.5 * self.kappa * (detF - 1)**2

    def _get_invariants(self, u, x):
        f0 = self.f0
        detF, trC = super()._get_invariants(u, x)

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

    def __call__(self, u, x):
        detF, trC, I4f0 = self._get_invariants(u, x)

        StrainEnergy = (self._get_compressibility(detF)
                      + self._get_passive_strain_energy(trC)
                      + self._get_active_strain_energy(detF, I4f0))

        return StrainEnergy
    
class GuccioneEnergyModel:
    def __init__(self, C, bf, bt, bfs, kappa=1E3):
        self.C = C
        self.bf = bf
        self.bt = bt
        self.bfs = bfs
        self.kappa = kappa

    def _get_passive_strain_energy(self, Q):
        return 0.5 * self.C * (torch.exp(Q) - 1)

    def _get_compressibility(self, detF):
        return  0.5 * self.kappa * (detF - 1)**2

    def _get_invariants(self, u, x):
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
        
        Cxx = Fxx*Fxx + Fyx*Fyx + Fzx*Fzx
        Cxy = Fxx*Fxy + Fyx*Fyy + Fzx*Fzy
        Cxz = Fxx*Fxz + Fyx*Fyz + Fzx*Fzz
        Cyx = Fxy*Fxx + Fyy*Fyx + Fzy*Fzx
        Cyy = Fxy*Fxy + Fyy*Fyy + Fzy*Fzy
        Cyz = Fxy*Fxz + Fyy*Fyz + Fzy*Fzz
        Czx = Fxz*Fxx + Fyz*Fyx + Fzz*Fzx
        Czy = Fxz*Fxy + Fyz*Fyy + Fzz*Fzy
        Czz = Fxz*Fxz + Fyz*Fyz + Fzz*Fzz

        # store C for use in active Guccione model
        self.Cxx = Cxx; self.Cxy = Cxy; self.Cxz = Cxz
        self.Cyx = Cyx; self.Cyy = Cyy; self.Cyz = Cyz
        self.Czx = Czx; self.Czy = Czy; self.Czz = Czz

        Exx = 0.5*(Cxx - 1)
        Exy = 0.5*(Cxy - 0)
        Exz = 0.5*(Cxz - 0)
        Eyx = 0.5*(Cyx - 0)
        Eyy = 0.5*(Cyy - 1)
        Eyz = 0.5*(Cyz - 0)
        Ezx = 0.5*(Czx - 0)
        Ezy = 0.5*(Czy - 0)
        Ezz = 0.5*(Czz - 1)

        # store E for use in tranversely isotropic Guccione model
        self.Exx = Exx; self.Exy = Exy; self.Exz = Exz
        self.Eyx = Eyx; self.Eyy = Eyy; self.Eyz = Eyz
        self.Ezx = Ezx; self.Ezy = Ezy; self.Ezz = Ezz

        detF = (Fxx * (Fyy * Fzz - Fyz * Fzy) 
              - Fxy * (Fyx * Fzz - Fyz * Fzx) 
              + Fxz * (Fyx * Fzy - Fyy * Fzx))
        
        Q = (self.bf*Exx**2
             + self.bt*(Eyy**2 + Ezz**2 + Eyz**2 + Ezy**2)
             + self.bfs*(Exy**2 + Eyx**2 + Exz**2 + Ezx**2))
        
        return detF, Q

    def __call__(self, u, x):
        detF, Q = self._get_invariants(u, x)

        StrainEnergy = (self._get_passive_strain_energy(Q)
                      + self._get_compressibility(detF))

        return StrainEnergy
    
class GuccioneTransverseEnergyModel(GuccioneEnergyModel):
    # Guccione energy model. Get source from verification paper!!!
    def __init__(self, C, bf, bt, bfs, kappa=1E3, 
                 f0=torch.tensor([1, 0, 0]),
                 s0=torch.tensor([0, 1, 0]),
                 n0=torch.tensor([0, 0, 1])):
        super().__init__(C, bf, bt, bfs, kappa)
        self.f0 = f0
        self.s0 = s0
        self.n0 = n0

    def _get_invariants(self, u, x):
        f0 = self.f0; s0 = self.s0; n0 = self.n0
        detF, _ = super()._get_invariants(u, x)
        
        Exx = self.Exx; Exy = self.Exy; Exz = self.Exz
        Eyx = self.Eyx; Eyy = self.Eyy; Eyz = self.Eyz
        Ezx = self.Ezx; Ezy = self.Ezy; Ezz = self.Ezz

        E11 = (f0[0] * (f0[0]*Exx + f0[1]*Eyx + f0[2]*Ezx)
             + f0[1] * (f0[0]*Exy + f0[1]*Eyy + f0[2]*Ezy)
             + f0[2] * (f0[0]*Exz + f0[1]*Eyz + f0[2]*Ezz))
        E12 = (s0[0] * (f0[0]*Exx + f0[1]*Eyx + f0[2]*Ezx)
             + s0[1] * (f0[0]*Exy + f0[1]*Eyy + f0[2]*Ezy)
             + s0[2] * (f0[0]*Exz + f0[1]*Eyz + f0[2]*Ezz))
        E13 = (n0[0] * (f0[0]*Exx + f0[1]*Eyx + f0[2]*Ezx)
             + n0[1] * (f0[0]*Exy + f0[1]*Eyy + f0[2]*Ezy)
             + n0[2] * (f0[0]*Exz + f0[1]*Eyz + f0[2]*Ezz))

        E21 = (f0[0] * (s0[0]*Exx + s0[1]*Eyx + s0[2]*Ezx)
             + f0[1] * (s0[0]*Exy + s0[1]*Eyy + s0[2]*Ezy)
             + f0[2] * (s0[0]*Exz + s0[1]*Eyz + s0[2]*Ezz))
        E22 = (s0[0] * (s0[0]*Exx + s0[1]*Eyx + s0[2]*Ezx)
             + s0[1] * (s0[0]*Exy + s0[1]*Eyy + s0[2]*Ezy)
             + s0[2] * (s0[0]*Exz + s0[1]*Eyz + s0[2]*Ezz))
        E23 = (n0[0] * (s0[0]*Exx + s0[1]*Eyx + s0[2]*Ezx)
             + n0[1] * (s0[0]*Exy + s0[1]*Eyy + s0[2]*Ezy)
             + n0[2] * (s0[0]*Exz + s0[1]*Eyz + s0[2]*Ezz))

        E31 = (f0[0] * (n0[0]*Exx + n0[1]*Eyx + n0[2]*Ezx)
             + f0[1] * (n0[0]*Exy + n0[1]*Eyy + n0[2]*Ezy)
             + f0[2] * (n0[0]*Exz + n0[1]*Eyz + n0[2]*Ezz))
        E32 = (s0[0] * (n0[0]*Exx + n0[1]*Eyx + n0[2]*Ezx)
             + s0[1] * (n0[0]*Exy + n0[1]*Eyy + n0[2]*Ezy)
             + s0[2] * (n0[0]*Exz + n0[1]*Eyz + n0[2]*Ezz))
        E33 = (n0[0] * (n0[0]*Exx + n0[1]*Eyx + n0[2]*Ezx)
             + n0[1] * (n0[0]*Exy + n0[1]*Eyy + n0[2]*Ezy)
             + n0[2] * (n0[0]*Exz + n0[1]*Eyz + n0[2]*Ezz))

        Q = (self.bf*E11**2
             + self.bt*(E22**2 + E33**2 + E23**2 + E32**2)
             + self.bfs*(E12**2 + E21**2 + E13**2 + E31**2))
        
        return detF, Q
    
class GuccioneTransverseActiveEnergyModel(GuccioneTransverseEnergyModel):
    def __init__(self, C, bf, bt, bfs, kappa=1E3, Ta=1.0,
             f0=torch.tensor([1, 0, 0]),
             s0=torch.tensor([0, 1, 0]),
             n0=torch.tensor([0, 0, 1])):
        super().__init__(C, bf, bt, bfs, kappa, f0, s0, n0)
        self.Ta = Ta

    def _get_active_strain_energy(self, detF, I4f0):
        return 0.5 * self.Ta / detF * (I4f0 - 1)
    
    def _get_invariants(self, u, x):
        f0 = self.f0
        detF, Q = super()._get_invariants(u, x)

        Cxx = self.Cxx; Cxy = self.Cxy; Cxz = self.Cxz
        Cyx = self.Cyx; Cyy = self.Cyy; Cyz = self.Cyz
        Czx = self.Czx; Czy = self.Czy; Czz = self.Czz

        I4f0 = (f0[0] * (f0[0]*Cxx + f0[1]*Cyx + f0[2]*Czx)
              + f0[1] * (f0[0]*Cxy + f0[1]*Cyy + f0[2]*Czy)
              + f0[2] * (f0[0]*Cxz + f0[1]*Cyz + f0[2]*Czz))

        return detF, Q, I4f0
    
    def __call__(self, u, x):
        detF, Q, I4f0 = self._get_invariants(u, x)

        StrainEnergy = (self._get_compressibility(detF)
                      + self._get_passive_strain_energy(Q)
                      + self._get_active_strain_energy(detF, I4f0))

        return StrainEnergy