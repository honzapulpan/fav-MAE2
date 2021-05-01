import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from IPython.display import display, Math
import ipywidgets as widgets


plt.rcParams['figure.figsize'] = [10, 10]


class wom():
    '''
    net_dim ... velikost sítě
    agents ... počet agentů, 0..1
    
    '''
    
    def __init__(self, 
                 net_dim = 10, 
                 agents = .8,
                 threshold = (.2,.1)):
        
        self.net_dim = net_dim
        
        # vytvořím pole, kde je daný počet agentů (agents) a všichni jsou S
        self.G = (np.random.rand(net_dim,net_dim) <= agents).astype(int)

        # pole s tresholdem T jednotlivých agentů        
        self.T = np.random.normal(threshold[0], threshold[1], (net_dim,net_dim))
        self.T = np.where(self.T>=0, self.T, 0)
        self.T = np.where(self.T<=1, self.T, 1)

        # pole s počtem agentů v 3-okolí
        self.G_acnt = np.zeros(self.G.shape)
        for n in range(1,4):
            #kříž
            self.G_acnt[:,n:] += self.G[:,:-n]
            self.G_acnt[:,:-n] += self.G[:,n:]
            self.G_acnt[n:,:] += self.G[:-n,:]
            self.G_acnt[:-n,:] += self.G[n:,:]
        for n in range(1,4):
            for m in range(1,4):
            #šikmo
                self.G_acnt[n:,m:] += self.G[:-n,:-m]
                self.G_acnt[n:,:-m] += self.G[:-n,m:]
                self.G_acnt[:-n,:-m] += self.G[n:,m:]
                self.G_acnt[:-n,m:] += self.G[n:,:-m]

        self.init_cmp()
        self.iters_no = 0

    
    def init_cmp(self):
        # https://www.w3schools.com/colors/colors_picker.asp
        vals = np.zeros((4, 4))
        vals[0] = np.array([.8,.8,.8,1]) # prázdná buňka
        vals[1] = np.array([57/255,115/255,172/255,1]) #S - modrá
        vals[2] = np.array([1,179/255,102/255,1])      #L - oranžová
        vals[3] = np.array([204/255,0,0,1])      #A - červená
        self.cmp = ListedColormap(vals)
        
    
    def _init_model(self, ic=False):
        
        if not(ic):
            self.G[2,2]=3
            self.G[3,2]=3
            self.G[4,2]=3
            self.G[2,3]=3
            self.G[3,3]=3
            self.G[4,3]=3

            self.G[2,5]=3
            self.G[3,5]=3
            self.G[4,5]=3
            self.G[5,5]=3
            
            self.G[7,5]=3
            
        else:
            for i in ic:
                for n in range(i[2]+1):
                    for m in range(i[2]+1):
                        self.G[i[0]+n,i[1]+m] = 3
            
        

        #chci nastavit počáteční podmínku tak, aby 
        #nastavit nějakej počet počátečních influencerů 
        
        #if ic.shape == self.G.shape:
        #    self.G = ic.astype(int)
        #    self.G = self.G.astype(int)
        #else:
        #    print('Počáteční podmínka neodpovídá velikostí modelu')
           
    def describe_model(self):
        plt.rcParams['figure.figsize'] = [10, 6]
        display(f'Průměrný threshold: {np.mean(self.T)}')
        plt.hist(self.T.flatten())
        plt.xlim(0, 1)
        plt.show()
        
    def _local_rule(self, p=1):

        # hledám v 3-okolí poměr počtu Influenced
        tmpG = np.zeros(self.G.shape)
        G_A = (self.G == 3).astype(int) # zajímají mě jen Influenced        
        for n in range(1,4):
            #kříž
            tmpG[:,n:] += G_A[:,:-n]
            tmpG[:,:-n] += G_A[:,n:]
            tmpG[n:,:] += G_A[:-n,:]
            tmpG[:-n,:] += G_A[n:,:]

            #šikmo
            tmpG[n:,n:] += G_A[:-n,:-n]
            tmpG[n:,:-n] += G_A[:-n,n:]
            tmpG[:-n,:-n] += G_A[n:,n:]
            tmpG[:-n,n:] += G_A[n:,:-n]

        # tmpG je teď poměr počtu Influenced ke počtu sousedů. 
        tmpG = tmpG / self.G_acnt
        
        # vemu jen tu část, kdy agnet je S=1, nebo L=2
        tmpG = np.where(self.G < 3, tmpG, 0)
        tmpG = np.where(self.G > 0, tmpG, 0)

        StoL = np.ceil(np.where(self.G==1, tmpG, 0))
        StoI = (self.T<np.where(self.G==1,tmpG,0)).astype(int)        
        LtoI = (self.T<np.where(self.G==2,tmpG,0)).astype(int)        
        self.G = self.G + StoL + StoI + LtoI
        
    def __plot_state(self, obj, labels=False):
        plt.rcParams['figure.figsize'] = [10, 10]
        
        fig, ax = plt.subplots()
        im = ax.imshow(obj, vmin=0, cmap=self.cmp, vmax=3)
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

            
    def simulate(self, ic=False, iters = 20):
        # !!!!! použít počáteční podmínku
        self._init_model(ic)
        self.iters_no = iters
        self.sim = np.empty((iters+1, self.net_dim, self.net_dim))
        
        self.sim[0] = self.G
        for i in range(1, iters+1):
            self._local_rule()
            self.sim[i] = self.G
  