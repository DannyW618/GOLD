# GOLD

This repository contains the implementation for the ICLR25 paper ["GOLD: Graph Out-of-Distribution Detection via Implicit Adversarial Latent Generation"](https://arxiv.org/abs/2502.05780).

## Intro

Despite graph neural networks' (GNNs) great success in modelling graph-structured data, out-of-distribution (OOD) test instances still pose a great challenge for current GNNs. We propose the GOLD framework for graph OOD detection, an implicit adversarial learning pipeline with synthetic OOD exposure without pre-trained models. The implicit adversarial training process employs a novel alternating optimisation framework by training: (1) a latent generative model to regularly imitate the in-distribution (ID) embeddings from an evolving GNN, and (2) a GNN encoder and an OOD detector to accurately classify ID data while increasing the energy divergence between the ID embeddings and the generative model's synthetic embeddings. This novel approach implicitly transforms the synthetic embeddings into pseudo-OOD instances relative to the ID data, effectively simulating exposure to OOD scenarios without auxiliary data.
<p align="center" width="100%">
<img width="800" alt="framework_gold" src="https://github.com/user-attachments/assets/5a3f0ea4-b8c9-422e-a540-91ff680c726b" />
</p>

## Dependence
- Python 3.8.0
- Cuda 12.1
- Pytorch 2.2.2
- ogb 1.3.3
- torch_geometric 2.0.3
- torch_sparse 0.6.18
  
## Usage
The dataset and splits utilised are downloaded automatically when running the training scripts. Alternatively, the datasets can be downloaded via running the ```load_dataset``` function in the ```gold/dataset.py``` file.

To execute the code, ``` cd ``` into the gold folder and run the ``` run_gold.sh ``` file.
```shell
cd gold
./run_gold.sh
```
Alternatively, you could also run the following command for individual datasets (i.e., Cora-structure):
```shell
python gold.py --backbone gcn --dataset cora --ood_type structure --mode detect --use_bn --use_prop --use_reg --m_in -5 --m_out -3 --device 0
```

We provide the model weights if you would like to perform inference or are having issues with executing the code. Run with the ``` --use_saved ``` keyword:
```shell
python gold.py --backbone gcn --dataset cora --ood_type structure --mode detect --use_bn --use_prop --device 0 --use_saved
```


## Acknowledgement

This repository takes credit from [GNNSafe](https://github.com/qitianwu/GraphOOD-GNNSafe/tree/main) for the energy-based detector backbone and [Neural Graph Generator](https://github.com/iakovosevdaimon/Neural-Graph-Generator/tree/main) for the diffusion process.

## Reference 
The bib ref for our paper is as follow:

```bibtex
@inproceedings{gold,
  title = {GOLD: Graph Out-of-Distribution Detection via Implicit Adversarial Latent Generation},
  author = {Danny Wang and Ruihong Qiu and Guangdong Bai and Zi Huang},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year = {2025}
  }
```
