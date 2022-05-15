#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  12, 2022

@author: rmojgani

PINN on Polar Coordinate
"""
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import time
import numpy as np
import matplotlib.pyplot as plt
# Packages
try:
    from tqdm import tqdm
except:
    import os
    os.system('!pip install tqdm')
    from tqdm import tqdm
            
class PINN_POLAR(nn.Module):
    # =========================================================================
    def __init__(self, N, M, HIDDEN, DEEPu, DEEPx, C, NU0, EQN_TYPE, IF_PLOT):
        # print('-----------------------------------')
        # print('PINN on Polar Coordinate')
        # print('-----------------------------------')

        # Variables
        self.N = N
        self.M = M
        self.HIDDEN = HIDDEN
        self.DEEPu = DEEPu
        # self.DEEPx = DEEPx
        self.c = C
        self.nu = NU0
        self.EQN_TYPE = EQN_TYPE
        # Settings
        self.IF_PLOT = IF_PLOT

        input  = 3 # xcos, xsin, t
        output = 1 # u

        # Initialize
        super().__init__()
        self.input = (nn.Linear(input, HIDDEN))
#        torch.nn.init.normal_(self.input.weight, mean=0, std =0.01)
        self.hidden1 = (nn.Linear(HIDDEN, HIDDEN))
#        torch.nn.init.normal_(self.hidden1.weight, mean=0, std =0.01)
        self.hidden2 = (nn.Linear(HIDDEN, HIDDEN))
#        torch.nn.init.normal_(self.hidden2.weight, mean=0, std =0.01)
        self.hidden3 = (nn.Linear(HIDDEN, HIDDEN))
##        torch.nn.init.normal_(self.hidden3.weight, mean=0, std =0.01)
        self.hidden4 = (nn.Linear(HIDDEN, HIDDEN))
##        torch.nn.init.normal_(self.hidden4.weight, mean=0, std =0.01)
        self.hidden5 = (nn.Linear(HIDDEN, HIDDEN))
#        torch.nn.init.normal_(self.hidden5.weight, mean=0, std =0.01)
        self.output = nn.Linear(HIDDEN, output)
#        torch.nn.init.normal_(self.output.weight, mean=0, std =0.01)

        # hidden2 = HIDDEN
        # self.hiddenae = (nn.Linear(HIDDEN, hidden2))
        # self.hiddennnn = (nn.Linear(hidden2, hidden2))
        # self.outpuaetttt = nn.Linear(hidden2, HIDDEN)

        # self.output2 = nn.Linear(HIDDEN, 2)
    # =========================================================================
#     def rx_func(self, x_t, u, u_x, u_xx, u_t, c):
        
#         if self.EQN_TYPE == 'convection':
#             return x_t - c
#         elif self.EQN_TYPE == 'Burgers':
#             return x_t - u
#         elif self.EQN_TYPE == 'reaction':
#             return x_t - c*u*(1-u)
        
# #    if self.EQN_TYPE == 'convection':
# #        def rx_func(self, x_t, u, u_x, u_xx, u_t, c):
# #            return x_t - c
# #    elif self.EQN_TYPE == 'Burgers':
# #        def rx_func(self, x_t, u, u_x, u_xx, u_t, c):
# #            return x_t - u
# #    elif self.EQN_TYPE == 'reaction':
# #        def rx_func(self, x_t, u, u_x, u_xx, u_t, c):
# #            return x_t - c*u*(1-u)
    # =========================================================================
    def forward(self, x, t, xx, tt):
        """ u = NNu(x,t) """
        """ xL = NNx(x,t) """
        
        xtheta = torch.cos(x)
        ytheta = torch.sin(x)

        x0 = torch.cat((xtheta, ytheta, t), axis = 1)
        x1 = torch.tanh(self.input(x0))# torch.tanh
        x2 = torch.tanh(self.hidden1(x1))
        x3 = torch.tanh(self.hidden2(x2))
#        x4 = torch.tanh(self.hidden2(x3))
#        x5 = torch.tanh(self.hidden2(x4))
#        u =            (self.output(x5)).cuda()
        if self.DEEPu == 3:
            u = (self.output(x3)).cuda()
        if self.DEEPu == 4:
            x4 = torch.tanh(self.hidden3(x3))
            u = (self.output(x4)).cuda()
        if self.DEEPu == 5:
            x4 = torch.tanh(self.hidden3(x3))
            x5 = torch.tanh(self.hidden4(x4))
            u = (self.output(x5)).cuda()
        if self.DEEPu == 6:
            x4 = torch.tanh(self.hidden3(x3))
            x5 = torch.tanh(self.hidden4(x4))
            x6 = torch.tanh(self.hidden5(x5))
            u = (self.output(x6)).cuda()

#         xL0 = torch.cat((xx,tt), axis = 1)
#         xL1 = torch.tanh(self.input(xL0))# torch.tanh
#         xL2 = torch.tanh(self.hidden1(xL1))
# #        xL3 = torch.tanh(self.hidden1(xL2))
#         xL =            (self.output(xL2)).cuda()

        return u, 0#, xL
    # =========================================================================
    def net_residual(self, x, t, xx, tt):
        """ dx/dt = F1() , du/dt = F2() 
        to do: generalize F2
            """
        u, u_x, u_xx, u_t, x_t = self.lib(x, t, xx, tt)
        # Evolution the characteristics lines
        # rx = self.rx_func(x_t, u, u_x, u_xx, u_t, self.c)
        # Evolution of the state on the characteristics lines
        # ru = u_t - self.nu*u_xx
        rx = 0
        if self.EQN_TYPE == 'convection':
            if self.nu == 0:
                ru = u_t + self.c * u_x
            else:
                ru = u_t + self.c * u_x - self.nu * u_xx
        elif self.EQN_TYPE == 'Burgers':
            ru = u_t + u * u_x - self.nu * u_xx
            
        return rx, ru
    # =========================================================================
    def lib(self, x, t, xx, tt):
        u, xL = self.forward(x, t, xx, tt)

        u_x = torch.autograd.grad(
            u, x, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True,
            allow_unused=True
        )[0]
        
        u_xx = torch.autograd.grad(
            u_x, x, 
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True,
            allow_unused=True
        )[0]
        
        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True,
            allow_unused=True
        )[0]
        
        # x_t = torch.autograd.grad(
        #     xL, tt,
        #     grad_outputs=torch.ones_like(xL),
        #     retain_graph=True,
        #     create_graph=True,
        #     allow_unused=True
        # )[0]
             
        return u, u_x, u_xx, u_t, 0#x_t
    # =========================================================================
    # def loss_rx(self, x, t, xx, tt):
    #     """ rx : dx/dt - F1() """
    #     rx_pred, ru_pred = self.net_residual(x, t, xx, tt)#rx_pred, ru_pred = self.net_residual(x, t, xx, tt)

    #     loss_rx = torch.norm( rx_pred, 2 )#torch.mean(rx_pred ** 2)

    #     return loss_rx 
    # =========================================================================
    def loss_ru(self, x, t, xx, tt):
        """ ru : du/dt - F2() """
        _, ru_pred = self.net_residual(x, t, xx, tt)

        loss_ru = torch.norm( ru_pred, 2 )#torch.mean(ru_pred ** 2)

        return loss_ru
    # =========================================================================
    def loss_x0u0(self, x, t, xx, tt, u0, x0):
        """ I.C. of x and u """
        N = self.N
        
        u, xL = self.forward(x, t, xx, tt)
        
        # loss_x0 =  torch.norm( xL[0:N,0].reshape(N,1) - x0, 2 )
        loss_u0 =  torch.norm( u[0:N,0].reshape(N,1) - u0, 2 )
        
        return loss_u0#loss_x0 + loss_u0
    # =========================================================================
    # def loss_pupx(self, x, t, xx, tt):
    #     """ Smoothing on u """
    #     N = self.N
    #     M = self.M
        
    #     u, xL = self.forward(x, t, xx, tt)
        
    #     uendminus1 = u.reshape(N,M)[-2,]
    #     uend = u.reshape(N,M)[-1,]
    #     ustart = u.reshape(N,M)[0,]
    #     ustartplus1 = u.reshape(N,M)[1,]

    #     xendminus1 = u.reshape(N,M)[-2,]
    #     xend = u.reshape(N,M)[-1,]
    #     xstart = u.reshape(N,M)[0,]
    #     xstartplus1 = u.reshape(N,M)[1,]



    #     loss_pupx = torch.norm(  (ustartplus1-ustart)/(xstartplus1-xstart)-(uend-uendminus1)/(xend-xendminus1) , 2)

        
    #     return loss_pupx
    # =========================================================================
#     def loss_ub(self, x, t, xx, tt, ub1, ub2):
#         """ B.C. of u """
#         N = self.N
#         M = self.M

#         u, _ = self.forward(x, t, xx, tt)

# #        loss_ub1 =  torch.norm( u.reshape(150,100)[0,:] - ub1, 2 )
# #        loss_ub2 =  torch.norm( u.reshape(150,100)[-1,:] - ub2, 2 )

#         loss_ubp = torch.norm( u.reshape(N,M)[0,] - u.reshape(N,M)[-1,], 2)
        
#         return loss_ubp#loss_ub1 + loss_ub2
    # =========================================================================
    def optimize_noclosure(self, x, t, 
                            u0, x0, ub1, ub2, 
                            NUM_EPOCHS, LR0, GAMMA,
                            optimizer):
        '''
        Parameters
        ----------
        optim: string,
                'ADAM', 'SGD'
        x : TYPE
            grid position.
        t : TYPE
            time of the solution.
        u0 : TYPE
            initial condition, u.
        x0 : TYPE
            initial condition, x.
        ub1 : TYPE
            boundary condition on x[0], u.
        ub2 : TYPE
            boundary condition on x[-1], u.
        NUM_EPOCHS : TYPE
            number of epochs.
        GAMMA : TYPE
            regularization weights.

        Returns
        -------
        None.
        '''
        IF_PLOT = self.IF_PLOT
        
        GAMMA_RX, GAMMA_RU, GAMMA_IC, GAMMA_BC = GAMMA

        if optimizer == 'SGD':
            optimizer = optim.SGD(self.parameters(), lr=LR0, momentum=0.9)
        elif optimizer == 'ADAM':
            optimizer = optim.Adam(self.parameters(), lr=LR0, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
                
        print('                 loss loss_rx, loss_ru, loss_x0u0, loss_ub')
        # for epoch in range(0, NUM_EPOCHS_ADAM):  # loop over the dataset multiple times
        for epoch in tqdm(range(0, NUM_EPOCHS)):
        
            # get the inputs; data is a list of [inputs, inputs, labels]
            x_batch, t_batch = x, t
            x_batch2, t_batch2 = x, t
            
            # zero the parameter gradients
            optimizer.zero_grad()
        
            # Forward
            u, xL = self.forward(x_batch.cuda(), t_batch.cuda(), 
                                 x_batch2.cuda(), t_batch2.cuda() )
            
            # Loss
            # Characteristics line eq.| R_x = 0
            # loss_rx = self.loss_rx(x_batch.cuda(), t_batch.cuda(), 
                                   # x_batch2.cuda(), t_batch2.cuda() )
            # State eq. on characteristics line | R_u = 0
            loss_ru = self.loss_ru(x_batch.cuda(), t_batch.cuda(), 
                                   x_batch.cuda(), t_batch.cuda() )
            # IC on u and x | u(x,0) = u0, x(x,0) = x0
            loss_x0u0 = self.loss_x0u0(x_batch.cuda(), t_batch.cuda(), 
                                        x_batch.cuda(), t_batch.cuda(),
                                        u0.cuda(), x0.cuda() )
            # BC on u | u_Left = L_right
            # loss_ub = self.loss_ub(x_batch.cuda(), t_batch.cuda(), 
            #                        x_batch.cuda(), t_batch.cuda(), 
            #                        ub1.cuda(), ub2.cuda() )
            # BC on x
            # .... check if necessary
            
            # BC on ∂u/∂x
            # loss_pupx = self.loss_pupx(x_batch.cuda(), t_batch.cuda(), 
                                       # x_batch.cuda(), t_batch.cuda() )
            # Total loss
            loss = GAMMA_RU*loss_ru + GAMMA_IC*loss_x0u0

            # Backprop
            loss.backward()
            # Optimize
            optimizer.step()

            # Print loss
            if epoch % int(NUM_EPOCHS/10) == 0:# print every with every 10% progress
                print('[%d] loss: %.2e %.2e %.2e %.2e %.2e' %
                      (epoch + 1, loss, 0, loss_ru, loss_x0u0, 0))
                if IF_PLOT:
                    self.myplot(x, t, u0, x0, epoch)
                
    # =========================================================================
    def optimize_LBFGS(self, x, t, u0, x0, ub1, ub2, NUM_EPOCHS, LR0, GAMMA):
        """
        LBFGS
        https://soham.dev/posts/linear-regression-pytorch/
        https://johaupt.github.io/python/pytorch/neural%20network/optimization/pytorch_lbfgs.html
        """
        IF_PLOT = self.IF_PLOT

        GAMMA_RX, GAMMA_RU, GAMMA_IC, GAMMA_BC = GAMMA

        optimizer = optim.LBFGS(self.parameters(), lr=LR0)

        #for epoch in range(0, num_epochs):  # loop over the dataset multiple times
        for epoch in tqdm(range(0, NUM_EPOCHS)):
        
            x_batch, t_batch = x, t
            x_batch2, t_batch2 = x, t
        
            ### Add the closure function to calculate the gradient.
            def closure():
                if torch.is_grad_enabled():
                  optimizer.zero_grad()
                  
                # Forward
                u, xL = self.forward(x_batch.cuda(), t_batch.cuda(), 
                                     x_batch2.cuda(), t_batch2.cuda() )
                
                # Loss
                # Characteristics line eq.
                # loss_rx = self.loss_rx(x_batch.cuda(), t_batch.cuda(),
                                       # x_batch2.cuda(), t_batch2.cuda() )
                # State eq. on characteristics line
                loss_ru = self.loss_ru(x_batch.cuda(), t_batch.cuda(), 
                                       x_batch.cuda(), t_batch.cuda() )
                # IC
                loss_x0u0 = self.loss_x0u0(x_batch.cuda(), t_batch.cuda(), 
                                            x_batch.cuda(), t_batch.cuda(), 
                                            u0.cuda(), x0.cuda() )
                # BC on u
                # loss_ub = self.loss_ub(x_batch.cuda(), t_batch.cuda(), 
                                       # x_batch.cuda(), t_batch.cuda(), 
                                       # ub1.cuda(), ub2.cuda() )
                # BC on x
                # .... check if necessary 
                
                # BC on ∂u/∂x
                # loss_pupx = self.loss_pupx(x_batch.cuda(), t_batch.cuda(), 
                                           # x_batch.cuda(), t_batch.cuda() )
                
                # Total loss    
                loss = GAMMA_RU*loss_ru + GAMMA_IC*loss_x0u0

                if loss.requires_grad:
                    loss.backward(retain_graph=True)
          
                return loss
                    
            optimizer.step(closure)
        
            # calculate the loss again for monitoring
            u, xL = self.forward(x_batch.cuda(), t_batch.cuda(), 
                                 x_batch2.cuda(), t_batch2.cuda() )
            loss = closure()
            
            # Print loss
            if epoch % int(NUM_EPOCHS/10) == 0:# print every with every 10% progress
                print('[%d] loss:  %.2e' %
                      (epoch + 1, loss))
                if IF_PLOT:
                    self.myplot(x, t, u0, x0, epoch)
    # =========================================================================
    def myplot(self, x, t, u0, x0, epoch):
        from pathlib import Path
        train_path = '../save/train'
        Path(train_path).mkdir(parents=True, exist_ok=True)

        N = self.N
        M = self.M
        xmax = 6.283185307179586
        xmin = 0
        X = np.linspace(xmin,xmax,N,endpoint=True).reshape((N,1))#np.arange(a, b, (b-a)/N).reshape((N,1))

        u, xL = self.forward(x.cuda(), t.cuda(),
                     x.cuda(), t.cuda() )

        us = u.detach().cpu().numpy().reshape(M,N).T
        xLs = x.detach().cpu().numpy().reshape(M,N).T
        xLs_wrap = xLs#%(xmax)
        #%%
        i_plot = np.linspace(0,M-1,4).astype(int)

        plt.figure(figsize=(12, 3))
        print(i_plot)
        plt.subplot(1,3,1)
        plt.plot(X, u0.detach().cpu().numpy()[:], label='truth')  
        plt.ylim([-1, 1])
        plt.title(r'$u_0$')
        plt.grid(color='r', alpha=0.25, linestyle='--', linewidth=2)
        plt.xlim([xmin,xmax])
        plt.subplot(1,3,2)
        # plt.plot(X, u_vals[:,i_plot],label='Exact')
        plt.ylim([-1, 1])
        plt.title('Exact')
        plt.grid(color='r', alpha=0.25, linestyle='--', linewidth=2)
        plt.xlim([xmin,xmax])

        plt.subplot(1,3,3)
        plt.plot(xLs_wrap[:,i_plot], us[:,i_plot],'.',label='nn')
        #plt.ylim([-1, 1])
        plt.title('LPINN')
        #plt.xlim([xmin,xmax])
        plt.grid(color='r', alpha=0.25, linestyle='--', linewidth=2)

        plt.savefig(train_path+'/PINN'+str(epoch)+'.png')
    #%%
    def loss_total(self,  x, t, 
                        u0, x0,
                        GAMMA):
    
        x_batch, t_batch = x, t
        GAMMA_RX, GAMMA_RU, GAMMA_IC, GAMMA_BC = GAMMA

        # Loss
        # State eq. | R_u = 0
        loss_ru = self.loss_ru(x_batch.cuda(), t_batch.cuda(), 
                               x_batch.cuda(), t_batch.cuda() )
        # IC on u and x | u(x,0) = u0, x(x,0) = x0
        loss_x0u0 = self.loss_x0u0(x_batch.cuda(), t_batch.cuda(), 
                                    x_batch.cuda(), t_batch.cuda(),
                                    u0.cuda(), x0.cuda() )

        # Total loss
        loss = GAMMA_RU*loss_ru + GAMMA_IC*loss_x0u0

        return loss