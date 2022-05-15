# -*- coding: utf-8 -*-
"""
Rambod Mojgani
PINN_Eq_Discovery_small.ipynb
"""
from LPINNarg import LPINNarg
args = LPINNarg()

locals().update(vars(args))
print(args)
GAMMA = [GAMMA_RX, GAMMA_RU, GAMMA_IC, GAMMA_BC]

IF_TEST = False
if IF_TEST:
    IF_LOCAL = True

    # NUM_EPOCHS_ADAM = int(1e6)
    NUM_EPOCHS_ADAM = int(1e4)
    # NUM_EPOCHS_ADAM = int(1e6/2)
    # NUM_EPOCHS_ADAM = int(2*1e6)

    NU0 = 0.01
    N = 512
    # NUM_EPOCHS_ADAM =  int(1e3)
    # NUM_EPOCHS_ADAM = 500000#0#0
    NUM_EPOCHS_BFGS = 0
    GAMMA_BC = 10
    C = 50.0

    NET_TYPE ='LPINN_POLAR'# choices=['LPINN_POLAR', 'PINN_POLAR']
    U0_TYPE = 'sin'#choices=['exp', 'gauss', 'sin', 'bell','sinpi','sin(x)']
    EQN_TYPE = 'Burgers'#choices=['Burgers', 'convection', 'reaction_diffusion', 'reaction'],
    to = 1.0
    # HIDDEN = 10
    # DEEPu = 6
    if NET_TYPE[-5:]=='POLAR':# For Polar
        GAMMA = [GAMMA_RX, GAMMA_RU, GAMMA_IC, 0]
        
    # NU0 = 0.0
    SEED = 3
#%%
import numpy as np
import torch
#from google.colab import drive
#drive.mount('/content/drive')
if IF_LOCAL:
    import matplotlib.pyplot as plt
    plt.rcParams['image.cmap'] = 'bwr_r'

#%% SEED
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
#%%
if NET_TYPE =='LPINN_POLAR': # Lagrangian PINN
    from mynet_POLAR import LPINN_POLAR as PINN
elif NET_TYPE =='PINN_POLAR': # Traditional PINN
    from mynet_POLAR0 import PINN_POLAR as PINN

if U0_TYPE=='sin' or U0_TYPE=='bell' or U0_TYPE=='sin(x)':
    xmin = 0.0
    xmax = 2*np.pi
elif U0_TYPE=='gauss':
    xmin = 0.0
    xmax = 1.5
elif U0_TYPE=='sinpi':
    xmin = 0.0
    xmax = 2.0
else:     
    print('TO BE ADDED')
    stop

ti = 0
# batch_size=10
#%% Save path
from pathlib import Path
folder_path='../save/'+NET_TYPE+'/'+EQN_TYPE+'/'+U0_TYPE+'/'
file_name = EQN_TYPE+'_weights_rxu_'+U0_TYPE+'_C'+str(C)+'_N'+str(HIDDEN)+'_l'+str(DEEPx)+\
                    'x'+str(DEEPu)+'_nu'+str(NU0)+\
                    '_maxSGS'+str(NUM_EPOCHS_SGD)+'_maxBFGS'+str(NUM_EPOCHS_BFGS)+'_maxADAM'+str(NUM_EPOCHS_ADAM)
Path(folder_path).mkdir(parents=True, exist_ok=True)
#%% x-t Grid
X = np.linspace(xmin,xmax,N,endpoint=True).reshape((N,1))#np.arange(a, b, (b-a)/N).reshape((N,1))
x_old=np.tile(X,(M,1))

T= np.linspace(ti,to,M,endpoint=True).reshape((M,1))#np.arange(ti,to,(to-ti)/M).reshape((M,1))
t_old=np.zeros([M*N,1])
count=0
for i in range (0,M):
  t_old[count:count+N]=T[i]  #### I've changed it to count+N
  count=count+N
#%% Initial condition
#u_truth_long = u_truth_long[:,np.linspace(0,200000-2,M).astype(int)]
if U0_TYPE == 'exp':
    u0 =  np.exp( -(X-xmax/3.0)**2/(0.25**2))
elif U0_TYPE == 'gauss':
    # u0 = np.exp(- (X-np.pi)**2/(2*(np.pi/4)**2)) + 0.0*C # Amir
    u0 = 0.8 + 0.5 * np.exp((-(X-0.3)**2)/(0.1**2)) # Rambod, arXiv: 1701.04343
elif U0_TYPE == 'bell':
    lmbd0 = np.pi+3*np.pi/2
    tht0 = 0
    h00 = 1000
    R = 1.0/3.0
    H = 1000
    
    r =  lambda lmbd, tht: np.arccos(np.sin(tht0)*np.sin(tht)+np.cos(tht0)*np.cos(tht)*np.cos(lmbd-lmbd0))
    h0 = lambda lmbd, tht: h00/2.0*(1+np.cos(np.pi*r(lmbd,tht)/R))*(r(lmbd,tht)<R)/H
    
    thetam = np.linspace(-np.pi/2,np.pi/2,M,endpoint=True).reshape((M,1))
    lambdam = np.pi+np.linspace(-np.pi,np.pi,N,endpoint=True).reshape((N,1))
    xv, yv = np.meshgrid(lambdam, thetam, sparse=True) 
    myh = h0(xv,yv)
    u0 = myh[50,:]
elif U0_TYPE == 'sinpi':
    u0 = np.sin(X*np.pi)
elif U0_TYPE == 'sin(x)':
    u0 = np.sin(X*2*np.pi/(xmax-xmin))
elif U0_TYPE == 'sin':
    u0 = 1.0*np.sin(X*2*np.pi/(xmax-xmin)) + C
    # u0 = 2.0*np.sin(X*2*np.pi/(xmax-xmin)) + C
#%% Initial condition plot
if IF_LOCAL:
    plt.plot(X, u0); plt.show()
#%% grid for pytorch
u0 = torch.tensor(u0,requires_grad=True)
x0 = torch.tensor(X ,requires_grad=True)

x=x_old#[idx,:]
t=t_old#[idx,:]

x = torch.from_numpy(x).float()
t = torch.from_numpy(t).float()

x = torch.tensor(x,requires_grad=True)
t = torch.tensor(t,requires_grad=True)

# print(np.shape(x))
# print(np.shape(t))
#%% Select device cuda/cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#%% Initilize pytorch model
net = PINN(N, M, HIDDEN, DEEPu, DEEPx, C, NU0, EQN_TYPE, IF_PLOT_TRAIN).to(device)#
net = net.to(device)# net = net.cuda()
#%% Number of network parameters
netnumel = 0
for var_name in  net.state_dict():
    netnumel += torch.numel(net.state_dict()[var_name])
print('network parameters / size of prediction=', netnumel/(N*M))
#%% Load saved weigths
if IF_LOADNET:
    net.load_state_dict(torch.load(folder_path+file_name+'.pth'))
#%%
ub1 = 0
ub2 = 0
#%% Train network: ADAM
if NUM_EPOCHS_ADAM != 0:
    net.optimize_noclosure(x, t, u0, x0, ub1, ub2, NUM_EPOCHS_ADAM, LR0, GAMMA, 'ADAM')
#%% Train network: SGD
if NUM_EPOCHS_SGD != 0:
    net.optimize_noclosure(x, t, u0, x0, ub1, ub2, NUM_EPOCHS_SGD, LR0, GAMMA, 'SGD')
#%% Train network: LBFGS
if NUM_EPOCHS_BFGS != 0:
    net.optimize_LBFGS(x, t, u0, x0, ub1, ub2, NUM_EPOCHS_BFGS, LR0, GAMMA)
#%% Save weights
if IF_SAVENET:
    torch.save(net.state_dict(),  folder_path+file_name+'.pth')
#%% The forcast grid
# M = 100
# to = 1
X = np.linspace(xmin,xmax,N,endpoint=True).reshape((N,1))#np.arange(a, b, (b-a)/N).reshape((N,1))
x_old=np.tile(X,(M,1))

T= np.linspace(ti,to,M,endpoint=True).reshape((M,1))#np.arange(ti,to,(to-ti)/M).reshape((M,1))
t_old=np.zeros([M*N,1])
count=0
for i in range (0,M):
  t_old[count:count+N]=T[i]  #### I've changed it to count+N
  count=count+N
  
xx = torch.from_numpy(x_old).float()
tt = torch.from_numpy(t_old).float()
xx = torch.tensor(xx,requires_grad=True)
#%% Predict
u, xL = net(xx.cuda(), tt.cuda(),
             xx.cuda(), tt.cuda() )
us = u.detach().cpu().numpy().reshape(M,N).T

if IF_LOCAL:
    fig = plt.figure(figsize=(4,3))
    
    ax = fig.add_subplot()
    if U0_TYPE == 'sin':
        VMIN, VMAX = C-1.0, C+1.0
    else:
        VMIN, VMAX = -1.0, 1.0

    plt.pcolor(xx.detach().cpu().numpy().reshape(M,N), 
                tt.detach().cpu().numpy().reshape(M,N), 
                us.T,vmin=VMIN,vmax=VMAX);plt.colorbar()
    
    plt.title(file_name)

if NET_TYPE == 'LPINN_POLAR':
    xLs = xL.detach().cpu().numpy().reshape(M,N).T
    xLs_wrap = xLs%(xmax)
    xLs_wrap = ((xLs-xmin)%(xmax-xmin))+xmin
else:
    xLs = xx.detach().cpu().numpy().reshape(M,N).T
    xLs_wrap = xLs
#%% Interpolate to stationary grid
import sys
from os.path import abspath, dirname
sys.path.insert(1, '../utils/')
if  EQN_TYPE == 'Burgers':
    from myburgers import myburgers
    u_vals = myburgers(NU0, C, N, 20*M, tmax = 1.0)
    i_save = np.linspace(0,20*M-1,M).astype(int)
    u_vals = u_vals[:,i_save]
else:
    import sys
    from os.path import abspath, dirname

    from systems_pbc import convection_diffusion
        
    u_vals = convection_diffusion('sin(x)', NU0, C, 0, N, M).reshape(M,N).T
    
    
from scipy import interpolate
# from scipy.interpolate import pchip_interpolate as interpolate

u_interpolated = np.zeros_like(us)
for tcount in range(M):
    f = interpolate.interp1d(xLs_wrap[:,tcount], us[:,tcount],
                              axis=0, 
                              kind='linear',#'linear', 'cubic', 'quadratic'
                              fill_value="extrapolate"#fill_value=(xLs_wrap[0,tcount], xLs_wrap[-1,tcount])
                              )
    u_interpolated[:,tcount] = f(X[:,0])
    # u_interpolated[1:-1,tcount] = interpolate(xLs_wrap[1:-1,tcount], us[1:-1,tcount], X[1:-1,0])

cut =2
if NET_TYPE=='LPINN_POLAR':
    du = u_vals[cut:-cut,:] - u_interpolated[cut:-cut,:]
elif NET_TYPE=='PINN_POLAR':
    du = u_vals[cut:-cut,:] - us[cut:-cut,:]

du_2 = np.linalg.norm(du,2)
du_fro = np.linalg.norm(du,'fro')
#%%
# print('<|------------ Error Table ------------|>')
# print('norm2', C, du_2, np.linalg.norm( u_vals[cut:-cut,:],  2  ) )
# print('normf', C, du_fro, np.linalg.norm( u_vals[cut:-cut,:],'fro') )
# print('-----------')
print('norm2', C, du_2, np.linalg.norm( u_vals[cut:-cut,:]-C,  2  ) )
# print('normf', C, du_fro, np.linalg.norm( u_vals[cut:-cut,:]-C,'fro') )
#%% Plot solution
if IF_LOCAL:
    i_plot = np.linspace(0,M-1,4).astype(int)
    
    plt.figure(figsize=(12, 3))
    print(i_plot)
    plt.subplot(1,3,1)
    plt.plot(X, u0.detach().cpu().numpy()[:], label='truth')  
    plt.ylim([VMIN, VMAX])
    plt.title(r'$u_0$')
    plt.grid(color='r', alpha=0.25, linestyle='--', linewidth=2)
    plt.xlim([xmin,xmax])
    
    plt.subplot(1,3,2)
    plt.plot(X, u_vals[:,i_plot], label='Exact', color='black', alpha=0.5, linewidth='4')
    # plt.plot(xLs_wrap[:,i_plot], us[:,i_plot],'.',label='nn',linestyle = 'None')
    plt.plot([C*1/2/np.pi,C*1/2/np.pi],[28,31],'r')
    plt.ylim([VMIN, VMAX])
    plt.title('Exact')
    plt.grid(color='r', alpha=0.25, linestyle='--', linewidth=2)
    plt.xlim([xmin,xmax])
    
    plt.subplot(1,3,3)
    plt.plot(xLs_wrap[:,i_plot], us[:,i_plot],'.',label='nn')
    plt.ylim([VMIN, VMAX])
    plt.title(NET_TYPE)
    #plt.xlim([xmin,xmax])
    plt.grid(color='r', alpha=0.25, linestyle='--', linewidth=2)
    
    plt.savefig(folder_path+file_name+'f.png')
    plt.show()

#%% Plot grid
if IF_LOCAL:
    i_plot_coarse = np.linspace(0,N-1,int(N/5),endpoint=True).astype(int)
    i_plot_fine = np.linspace(0,N-1,10,endpoint=True).astype(int)
    
    plt.figure(figsize=(15, 3))
    
    plt.subplot(1,3,1)
    plt.pcolor(xLs,cmap='bwr');
    plt.pcolor( np.diff(xLs.T).T, cmap = 'bwr', vmin = 0.95*(xmax-xmin)/(N-1)
                                               , vmax = 1.05*(xmax-xmin)/(N-1) );
    # plt.pcolor( np.diff(xLs.T).T)
    plt.colorbar()
    plt.title(r'$\Delta x$',fontsize=12)
    plt.xlabel(r'$N_t$',fontsize=12)
    plt.ylabel(r'$N_x$',fontsize=12)
    
    
    plt.subplot(1,3,2)
    plt.plot(T, xLs[i_plot_coarse, :].T,'-k',alpha=0.25)
    plt.plot(T, xLs[i_plot_fine, :].T,'k')
    plt.xlabel(r'$N_t$',fontsize=12)
    plt.ylabel(r'$x$',fontsize=12)
    plt.grid(color='r', alpha=0.25, linestyle='--', linewidth=2)
    # plt.ylim([-0.5,6.5])
    #
    ##%%
    #
    #plt.subplot(1,3,3)
    ##plt.plot(xLs[np.linspace(0,N-1,10,endpoint=True).astype(int),:].T%(2*np.pi),'k')
    ##plt.contourf(x_old.reshape(*us.shape), t_old.reshape(*us.shape), us, cmap='bwr');
    ##plt.contourf( xLs%(xmax), tt.detach().cpu().numpy().reshape(*us.T.shape,order='C').T, us , cmap = 'bwr')#, vmin = 0.9*(xmax-xmin)/N, vmax = 1.1*(xmax-xmin)/N );
    #plt.contourf(xLs_wrap,  t_old.reshape(*us.T.shape).T, u_wrap)
    #plt.xlabel(r'$N_t$',fontsize=12)
    #plt.ylabel(r'$N_x$',fontsize=12)
    
    
    plt.savefig(folder_path+file_name+'c.png')
    
    print('mean % off : ', 100*(np.mean(xLs[-1,:]-xLs[0,:])-xmax)/xmax, ', std: ',np.std(xLs[-1,:]-xLs[0,:]))
#%% Wrap u
u_wrap = us.copy()
ind_min = np.argmin(np.diff(xLs_wrap).T,axis=1)
for tcount in range(0,M-1):
    u_wrap[:,tcount] = np.roll(us[:,tcount], -ind_min[tcount])
#    plt.plot(xLs_wrap[:,tcount], u_wrap[:,tcount])
#    plt.show()
#%% Plot interpolated u
if IF_LOCAL:
    plt.figure()

    if NET_TYPE=='LPINN_POLAR':
        plt.pcolor(xx.detach().cpu().numpy().reshape(M,N), 
                    tt.detach().cpu().numpy().reshape(M,N),
                    u_interpolated.T,
                    vmin=VMIN, vmax=VMAX);
    elif NET_TYPE=='PINN_POLAR':
        plt.pcolor(xx.detach().cpu().numpy().reshape(M,N), 
                    tt.detach().cpu().numpy().reshape(M,N),
                    us.T,
                    vmin=VMIN, vmax=VMAX);
        
    plt.colorbar()
    plt.xlabel(r'$x$',fontsize=12)
    plt.ylabel(r'$t$',fontsize=12)