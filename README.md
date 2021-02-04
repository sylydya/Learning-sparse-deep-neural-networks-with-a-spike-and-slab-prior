Extended-SGMCMC-for-BNN-with-spike-and-slab-prior
===============================================================
Applied extended SGMCMC method from paper Extended Stochastic Gradient MCMC for Large-Scale Bayesian Variable Selection (https://arxiv.org/abs/2002.02919) to high dimensional variable selection and network pruning

### Simulation:

Generate Data:
```{python}
python Generate_Data.py
```

Variable Selection:
```{python}
python Simulation_Regression.py --data_index 1
```

Structure Selection
```{python}
python Simulation_Structure.py --data_index 1
```
### Network Compression
MNIST Compression
```{python}
python mnist_300_100.py
```
    
CIFAR-10 Compression
```{python}
python cifar_run.py -depth 20 --Proposal_B 400 250 --lambdan 0.0001
python cifar_run.py -depth 20 --Proposal_B 500 500 --lambdan 0.001
python cifar_run.py -depth 32 --Proposal_B 400 180 --lambdan 0.0001
python cifar_run.py -depth 32 --Proposal_B 400 300 --lambdan 0.001
python cifar_run_vgg.py
```
CIFAR-10 Result, average over 3 runs. The first four lines denote the result using the model at last epoch. The last four line denot the result using Bayesian model average over models at the last 75 epochs.
Model Average | Model   |      Sparsity      | Accuracy  |
|----------|:-------------:|:-------------:|:-------------:|
|No   | ResNet20 | 9.88\%(0.08\%)  | 91.26(0.02) |
|No   | ResNet20 | 19.83\%(0.02\%) | 92.32(0.04) | 
|No   | ResNet32 | 8.77\%(0.12\%)  | 92.74(0.10) | 
|No   | ResNet32 | 4.99\%(0.06\%)  | 91.39(0.10) | 
|Yes  | ResNet20 | 9.65\%(0.05\%)  | 91.60(0.06) |
|Yes  | ResNet20 | 19.76\%(0.02\%) | 92.65(0.02) | 
|Yes  | ResNet32 | 8.69\%(0.16\%)  | 92.99(0.08) | 
|Yes  | ResNet32 | 4.89\%(0.09\%)  | 91.84(0.09) | 

