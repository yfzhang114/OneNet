# OneNet: Enhancing Time Series Forecasting Models under Concept Drift by Online Ensembling

This codebase is the official implementation of [`OneNet: Enhancing Time Series Forecasting Models under Concept Drift by Online Ensembling`]() (**NeurIPS 2023 poster**)


This codebase is mainly based on [FSNet](https://github.com/salesforce/fsnet).

## Requirements

- python == 3.7.3
- pytorch == 1.8.0
- matplotlib == 3.1.1
- numpy == 1.19.4
- pandas == 0.25.1
- scikit_learn == 0.21.3
- tqdm == 4.62.3
- einops == 0.4.0

## Benchmarking

### 1. Data preparation

We follow the same data formatting as the Informer repo (https://github.com/zhouhaoyi/Informer2020), which also hosts the raw data.
Please put all raw data (csv) files in the ```./data``` folder.

### 2. Run experiments

To replicate our results on the ETT, ECL, Traffic, and WTH datasets, run
```
sh run.sh
```

### 3.  Arguments


You can specify one of the above method via the ```--method``` argument.

**Dataset:** Our implementation currently supports the following datasets: Electricity Transformer - ETT (including ETTh1, ETTh2, ETTm1, and ETTm2), ECL, Traffic, and WTH. You can specify the dataset via the ```--data``` argument.

**Other arguments:** Other useful arguments for experiments are:
- ```--test_bsz```: batch size used for testing: must be set to **1** for online learning,
- ```--seq_len```: look-back windows' length, set to **60** by default,
- ```--pred_len```: forecast windows' length, set to **1** for online learning.

### 4.  Baselines

**Backbones:** Our implementation supports the following backbones in Table.1:

- patch: PatchTST for online time series forecasting
- fedformer: FedFormer for online time series forecasting
- dlinear: DLinear for online time series forecasting
- cross_former: Crossformer for online time series forecasting
- naive_time: The proposed Time-TCN for online time series forecasting
- naive_time: The proposed Time-TCN for online time series forecasting


**Ablations:** Our online learning and ensembling ablation baselines in Table.4:
- fsnet_plus_time: Simple averaging
- onenet_gate: Gating mechanism
- onenet_linear_regression: Linear Regression (LR)
- onenet_egd: Exponentiated Gradient Descent (EGD)
- onenet_weight: Reinforcement learning to learn the weight directly (RL-W)

**Algorithms:** Our implementation supports the following training strategies in Table.2,3:
- ogd: OGD training
- large: OGD training with a large backbone
- er: experience replay
- derpp: dark experience replay
- nomem: FSNET without the associative memory
- naive: FSNET without both the memory and adapter, directly trains the adaptation coefficients.
- fsnet: FSNet framework
- fsnet_time: Cross-Time FSNet
- onenet_minus: the proposed OneNet- in section 4
- onenet_tcn: the proposed OneNet with tcn backbone
- onenet_fsnet: the proposed OneNet 

## License

This source code is released under the MIT license, included [here](LICENSE).

### Citation 
If you find this repo useful, please consider citing: 
```
@inproceedings{
    anonymous2023onenet,
    title={OneNet: Enhancing Time Series Forecasting Models under Concept Drift by Online Ensembling},
    author={Anonymous},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=Q25wMXsaeZ}
}
```
