Bayesian Sparse Deep Neural Networks: Theory and Computation
===============================================================

### Installation

Requirements for Pytorch see [Pytorch](http://pytorch.org/) installation instructions

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
Variable Selection Baseline:
    ```
    python Spinn_Regression.py --data_index 1
    ```

##### Command For Running Structure Selection Experiment
Structure Selection
    ```
    python Simulation_Structure.py --data_index 1
    ```
Structure Selection Baseline:
    ```
    python Spinn_structure.py --data_index 1
    ```

##### Command For Running Real Data Experiment:
MNIST Compression
    ```
    python mnist_300_100.py
    ```
CIFAR Compression
    ```
    python cifar_run.py -depth 20 --Proposal_B 400 250 --lambdan 0.0001
    python cifar_run.py -depth 20 --Proposal_B 500 500 --lambdan 0.001
    python cifar_run.py -depth 32 --Proposal_B 400 180 --lambdan 0.0001
    python cifar_run.py -depth 32 --Proposal_B 400 300 --lambdan 0.001
    python cifar_run_vgg.py
    ```
