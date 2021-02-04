Extended-SGMCMC-for-BNN-with-spike-and-slab-prior
===============================================================
Applied extended SGMCMC method from paper Extended Stochastic Gradient MCMC for Large-Scale Bayesian Variable Selection (https://arxiv.org/abs/2002.02919) to high dimensional variable selection and network pruning

### Simulation:

Generate Data:
    ```
    python Generate_Data.py
    ```
##### Command For Running Variable Selection Experiment
Variable Selection:
    ```
    python Simulation_Regression.py --data_index 1
    ```

##### Command For Running Structure Selection Experiment
Structure Selection
    ```
    python Simulation_Structure.py --data_index 1
    ```

##### Command For Running Real Data Experiment:
MNIST Compression
    ```
    python mnist_300_100.py
    ```
CIFAR Compression
```{python}
python cifar_run.py -depth 20 --Proposal_B 400 250 --lambdan 0.0001
python cifar_run.py -depth 20 --Proposal_B 500 500 --lambdan 0.001
python cifar_run.py -depth 32 --Proposal_B 400 180 --lambdan 0.0001
python cifar_run.py -depth 32 --Proposal_B 400 300 --lambdan 0.001
python cifar_run_vgg.py
```
Model Average | Model   |      Sparsity      | Accuracy  |
|----------|:-------------:|:-------------:|:-------------:|
|No   | ResNet20 | 9.88\%(0.08\%)  | 91.26(0.02) |
|No   | ResNet20 | 19.83\%(0.02\%) | 92.32(0.04) | 
|No   | ResNet32 | 8.77\%(0.12\%)  | 92.74(0.10) | 
|No   | ResNet32 | 4.99\%(0.06\%)  | 91.39(0.10) | 
