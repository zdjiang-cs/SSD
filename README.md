Enhancing Decentralized Machine Learning with On-device Unlabeled Data
===
This repository is an official Pytorch implementation of our paper Enhancing Decentralized Machine Learning with On-device Unlabeled Data.


Introduction
---
<img src="https://github.com/zdjiang-cs/SSD/blob/main/training_module/image/SSD.png" width="500"><br>
How to exploit the unlabeled data residing on decentralized workers is a new challenge for decentralized machine learning (DML) and also is an understudied problem. This paper proposes a novel framework, called SSD, which addresses the problem of semi-supervised DML by adaptive neighbor selection.

As shown in Figure, left plot is the research scenario of semi-supervised DML. Each worker in the P2P network has both labeled and unlabeled data. Right plot is the illustration of the training process on worker i. Each round t consists of four phases: 
* Supervised learning
* Model exchange and aggregation
* Pseudo-label generation and selection
* Unsupervised learning

The above procedure repeats for T training rounds. Considering the pseudo-label quality and communication overhead, SSD enables each worker to select the neighbors with high-quality models and similar data distribution under communication resource constraints. Adaptive neighbor selection helps to generate high-confidence pseudo-labels for local unlabeled data and thus boosts the DML performance.


Requirements
---
Pytorch v1.6 <br>
CUDA v10.0 <br>
cuDNN v7.5.0 <br>

Organization of the code
---
* `client_module` is designed for each worker, including implementations of local training, model aggregation, and pseudo-label generation.<br>
* `communication_module` contains implementations of establishing communication links.<br>
* `training_module` contains implementations of parameter configuration, creating models and datasets.<br>
* `initialization.py` is designed to initialize training parameters, configure P2P ports, etc. <br>

Usage
---
In SSD, we conduct experiments on a hardware prototype system, which includes two types of embedded devices with heterogeneous capabilities. To reproduce experiments, please migrate `client_module` to embedded devices (e.g., NVIDIA Jetson TX2), and run the following command for the workers:<br>
```python
    python client.py
```
Please replace `CLIENT_IP` with your IP and then run the following command to initialize training:<br>
```python
    python initialization.py --model_type 'CNN' --dataset_type 'FashionMNIST' --epoch 200
```
