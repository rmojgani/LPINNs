# Example:
# bash local_bash.sh > LPINNconvnu0d00_4x2__quad.out
#
#for C in 2.0 5.0 10.0 20.0 30.0 40.0 50.0; do
NU0=0.01
DEEPu=5
IF_LOCAL=False
IF_PLOT_TRAIN=True

IF_LOADNET=False
IF_SAVENET=True

for C in 0.0 2.0 5.0 10.0 20.0 30.0 40.0 50.0 60.0 70.0; do
python3 code/LPINN.py --C=$C --NU0=$NU0 --DEEPu=$DEEPu --IF_PLOT_TRAIN=$IF_PLOT_TRAIN  --NUM_EPOCHS_ADAM=100
#--IF_LOCAL=$IF_LOCAL --IF_LOADNET=$IF_LOADNET --IF_SAVENET=$IF_SAVENET
done
