import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from IPython.display import display
import ipywidgets as widgets
from scipy.integrate import solve_ivp

plt.rcParams['figure.figsize'] = [10, 10]


class greenberg_hastings():
    '''
    net_dim ... velikost sítě
    a ... počet časových jednotek 
    g ... počet časových jednotek 
    '''
    
    def __init__(self, 
                 net_dim = 20, 
                 a = 2, 
                 g = 3):
        
        self.net_dim = net_dim
        self.a = a
        self.g = g
        self.e = a + g
        
        self.G = np.empty((net_dim, net_dim))
        
        self.init_cmp()
        
        self.iters_no = 0
    
    
    def init_cmp(self):
        N = self.a + self.g + 1
        
        vals = np.zeros((N, 4))
        vals[:,3] = 1
        vals[0] = np.array([0,80/256,0,1])

        vals[1:self.a+1, 0] = 1
        vals[1:self.a+1, 1] = np.linspace(53/256, 116/256, self.a)
        
        vals[self.a+1:, 0] = vals[self.a+1:, 2] = 10/256
        vals[self.a+1:, 1] = np.linspace(10/256, 60/256, N-self.a-1)
        
        self.cmp = ListedColormap(vals)
        
    
    def _init_model(self, ic):
        if ic.shape == self.G.shape:
            self.G = ic.astype(int)
            self.G = self.G.astype(int)
        else:
            print('Počáteční podmínka neodpovídá velikostí modelu')
           
        
    def _local_rule(self, p=1):
        # počet hořících sousedů ... 
        tmpG = np.zeros(self.G.shape)
        G_f = np.where(self.G <= self.a, self.G, 0) # f as on fire
        tmpG[:,1:] += (G_f[:,:-1] > 0).astype(int)
        tmpG[:,:-1] += (G_f[:,1:] > 0).astype(int)
        tmpG[1:,:] += (G_f[:-1,:] > 0).astype(int)
        tmpG[:-1,:] += (G_f[1:,:] > 0).astype(int)


        tmpG = (tmpG>0).astype(int)
        # ... mě zajímá jen u stromů které nehoří

        tmpG = (np.random.uniform(0., 1., self.G.shape) <= 1-(1-p)**tmpG).astype(int)
            
        tmpG = np.where(self.G==0, tmpG, 0) #

        # ke všem stromům, které hoří, nebo jsou už na prach, přičtu 1
        self.G += (self.G > 0).astype(int)
        # a teď zapálím všechny stromy které nehoří, ale mají hořícího souseda
        self.G += tmpG
        # nakonec stromy které se již vzpamatovaly uděláme záze "požáruhodné"
        self.G[self.G > self.e] = 0
        
    def _local_rule_moore(self, p=1):
        # počet hořících sousedů ... 
        tmpG = np.zeros(self.G.shape)
        G_f = np.where(self.G <= self.a, self.G, 0) # f as on fire
        tmpG[:,1:] += (G_f[:,:-1] > 0).astype(int)
        tmpG[1:,1:] += (G_f[:-1,:-1] > 0).astype(int)
        tmpG[:,:-1] += (G_f[:,1:] > 0).astype(int)
        tmpG[:-1,:-1] += (G_f[1:,1:] > 0).astype(int)
        tmpG[1:,:] += (G_f[:-1,:] > 0).astype(int)
        tmpG[1:,:-1] += (G_f[:-1,1:] > 0).astype(int)
        tmpG[:-1,:] += (G_f[1:,:] > 0).astype(int)
        tmpG[:-1,1:] += (G_f[1:,:-1] > 0).astype(int)        


        tmpG = (tmpG>0).astype(int)
        # ... mě zajímá jen u stromů které nehoří

        #if tp == 'stochastic':
        tmpG = (np.random.uniform(0., 1., self.G.shape) <= 1-(1-p)**tmpG).astype(int)
            
        tmpG = np.where(self.G==0, tmpG, 0) #

        # ke všem stromům, které hoří, nebo jsou už na prach, přičtu 1
        self.G += (self.G > 0).astype(int)
        # a teď zapálím všechny stromy které nehoří, ale mají hořícího souseda
        self.G += tmpG
        # nakonec stromy které se již vzpamatovaly uděláme záze "požáruhodné"
        self.G[self.G > self.e] = 0
        
    def __plot_state(self, obj, labels=False):
        plt.rcParams['figure.figsize'] = [10, 10]
        
        fig, ax = plt.subplots()
        #im = ax.imshow(harvest)
        im = ax.imshow(obj, cmap=self.cmp, vmin=0, vmax=self.e)
        plt.axis('off')
        
        if labels:
            for i in range(self.net_dim):
                for j in range(self.net_dim):
                    text = ax.text(j, i, str(int(obj[i, j])),
                                   ha="center", va="center", color="w")
        
        
        plt.show()
    
        
    def plot_sim_state(self, t, labels=False):
        # vykreslí stav v daném čase
        if self.iters_no == 0:
            print('Simulace jěště neproběhla.')
        self.__plot_state(self.sim[t], labels)

        
    def plot_sol(self, gh=True, sir=True):
        #def sirs_solve(beta, gamma, nu):
    
        def sirs_ode(t,ivs,params):
            beta, gamma, nu = params
            S,I,R = ivs
            N = S+I+R

            dS = -beta*S*I/N + nu*R
            dI = beta*S*I/N - gamma*I
            dR = gamma*I-nu*R

            return [dS,dI,dR]

        if gh & sir: 
            fig = plt.figure(figsize=(29,10))
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
        else:
            if gh:
                fig = plt.figure(figsize=(14,10))
                ax1 = fig.add_subplot(111)
            else:
                fig = plt.figure(figsize=(14,10))
                ax2 = fig.add_subplot(111)

                
        R = np.zeros(len(self.sim))
        X = np.zeros(len(self.sim))
        F = np.zeros(len(self.sim))
        for i in range(len(R)):
            R[i] = np.count_nonzero(self.sim[i] == 0)
            X[i] = np.count_nonzero((self.sim[i] > 0) & (self.sim[i] < self.a+1))
            F[i] = np.count_nonzero(self.sim[i] > self.a)

        if gh:
            splot = ax1.plot(R, color='green', linewidth=2, label='R')
            iplot = ax1.plot(X, color='orange', linewidth=2, label='X')
            rplot = ax1.plot(F, color='black', linewidth=2, label='F')

            ax1.set_xlabel("t")
            legend = ax1.legend(loc=5,bbox_to_anchor=(1.1,0.5))
            frame = legend.get_frame()
            frame.set_facecolor("white")

        if sir:
            # spojity SIRS
            params = [.3, 1/self.a, 1/self.g]
            # vykresleni reseni SIRS
            ivs = [R[0],X[0],F[0]]
            time = np.linspace(0,len(R),801)

            sirs_sol = solve_ivp(fun=lambda t, y: sirs_ode(t, y, params), 
                                 t_span=[min(time),max(time)], 
                                 y0=ivs, 
                                 t_eval=time)

            t_sol = sirs_sol['t']
            s_sol, i_sol, r_sol = sirs_sol['y']

            splot = ax2.plot(t_sol, s_sol, color='green', linewidth=2, label='S')
            iplot = ax2.plot(t_sol, i_sol, color='orange', linewidth=2, label='I')
            rplot = ax2.plot(t_sol, r_sol, color='black', linewidth=2, label='R')

            ax2.set_xlabel("t")
            legend = ax2.legend(loc=5,bbox_to_anchor=(1.1,0.5))
            frame = legend.get_frame()
            frame.set_facecolor("white")


        plt.show()
            
    def simulate(self, ic, iters_no = 20, p=1, ngbr='neumann'):
        self._init_model(ic)
        self.iters_no = iters_no
        self.sim = np.empty((iters_no+1, self.net_dim, self.net_dim))
        
        self.sim[0] = self.G
        for i in range(1, iters_no+1):
            if ngbr == 'neumann':
                self._local_rule(p=p)
            elif ngbr == 'moore':
                self._local_rule_moore(p=p)
            self.sim[i] = self.G
  
