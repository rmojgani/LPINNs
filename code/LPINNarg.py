#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 00:36:05 2022

@author: rmojgani
"""

def LPINNarg():
    import argparse
    parser = argparse.ArgumentParser(description='PINN of convection-dominated 1D flows on Lagrangian framework')

    # Case parameters
    parser.add_argument('--EQN_TYPE', type=str, default='convection',\
                        choices=['Burgers', 'convection', 'reaction_diffusion', 'reaction'], \
                            help='Equation type')
    parser.add_argument('--C', type=float, default='50.0', \
                        help='Convection/wave speed')
    parser.add_argument('--NU0', type=float, default='0.01', \
                        help='Viscosity')

    parser.add_argument('--U0_TYPE', type=str, default='sin',\
                        choices=['exp' , 'gauss', 'sin','bell','sin(x)'], \
                            help='Initial condition case')
    parser.add_argument('--to', type=float, default=1.0, help='t_{max}')
        
    # Architecture parameters
    parser.add_argument('--NET_TYPE', type=str, default='LPINN_POLAR',\
                        choices=['LPINN_POLAR', 'PINN_POLAR'], \
                            help='Network architecture')
    parser.add_argument('--DEEPu', type=int, default=5, choices=range(1, 10), \
                        help='u-Network deep layers')
    parser.add_argument('--DEEPx', type=int, default=2, choices=range(1, 10), \
                        help='x-Network deep layers')
    parser.add_argument('--HIDDEN', type=int, default=50, choices=range(1, 10), \
                        help='Nodes in deep layers')
    parser.add_argument('--SEED', type=int, default=0, help='Pseudo-random seed')


    # Data parameters
    parser.add_argument('--N', type=int, default=256, choices=range(200, 513), \
                        help='Space discretization size')
    parser.add_argument('--M', type=int, default=100, choices=range(100, 1000), \
                        help='Time discretization size')

    # Optimizer parameters
    parser.add_argument('--NUM_EPOCHS_ADAM', type=int, default=int(1e6), \
                        help='Number of epoch, ADAM')
    parser.add_argument('--NUM_EPOCHS_SGD', type=int, default=int(0), \
                        help='Number of epoch, SGD')
    parser.add_argument('--NUM_EPOCHS_BFGS', type=int, default=int(1e2), \
                        help='Number of epoch, BFGS')
    parser.add_argument('--LR0', type=float, default=0.01, \
                        help='[Initial] learning rate')

    parser.add_argument('--GAMMA_RX', type=float, default=10.0, help='GAMMA_RX')
    parser.add_argument('--GAMMA_RU', type=float, default=1.0, help='GAMMA_RU')
    parser.add_argument('--GAMMA_IC', type=float, default=1000.0, help='GAMMA_IC')
    parser.add_argument('--GAMMA_BC', type=float, default=10.0, help='GAMMA_BC')
    
    
    # Report parameters
    parser.add_argument('--IF_LOCAL', type=bool, default=False,\
                        choices=[False, True], help='Code is run from cmd?\
                            (display plots)')
    parser.add_argument('--IF_PLOT_TRAIN', type=bool, default=False,\
                        choices=[False, True], help='Plot training phase?\
                            (saves plots during training)')
    parser.add_argument('--IF_LOADNET', type=bool, default=False,\
                        choices=[False, True], help='Load saved weghts?')
    parser.add_argument('--IF_SAVENET', type=bool, default=True,\
                        choices=[False, True], help='Save training weights?')

    return parser.parse_args()
