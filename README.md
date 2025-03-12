# Code for Progressive Goal-oriented Communications for Reinforcement Learning Control over Multi-tier Computing Systems
This repository is the implementation of MLVE. Our implementation of SAC is based on [SAC+AE](https://github.com/denisyarats/pytorch_sac_ae) by Denis Yarats.
## Train the MLVE codec (i.e., discrete version)
To train the MLVE codec on the task from image-based observations, navigate to the directory: [mlve/mlve-continuous/pytorch_sac_mlve_discrete/](https://github.com/dezhaochen/mlve/tree/main/mlve-discrete/pytorch_sac_mlve_discrete). The file contains the following command, which you can modify to try different environments / hyperparamters. ```source train.sh```
```
python train.py \
    --domain_name cartpole \
    --task_name swingup \
    --encoder_type pixel \
    --decoder_type pixel \
    --action_repeat 4 \
    --lambdaR 1e-10 1e-4 1e-4 \
    --lambdaD 1e-6 \
    --lambdaE 1e-5 1e-4 \
    --qt 1000 100 1 \
    --KLl 22 2 \
    --work_dir ./logsave/ \
    --seed 1
```
In your console, you should see printouts that look like:
```
| train | E: 1 | S: 250 | D: 60.1 s | R: 0.0000 | BR: 0.0000 | ALOSS: 0.0000 | CLOSS: 0.0000 | DLOSS: 0.0000 | rec: 0.0000 | kl1: 0.0000 | kl2: 0.0000 | aux: 0.0000 | z1_bpp: 0.0000 | z2_bpp: 0.0000 | z3_bpp: 0.0000 | z2_LR-scl: 0.0000 | z3_LR-scl: 0.0000 | conloss: 0.0000
```
A training entry decodes as:
```
train - training episode
E - total number of episodes 
S - total number of environment steps
D - duration in seconds to train 1 episode
R - episode reward
BR - average reward of sampled batch
ALOSS - average loss of actor
CLOSS - average loss of critic
DLOSS - average loss of decoder
rec - average loss of reconstruction
kl1/kl2 - average KL value at each layer
aux - average auxiliary loss of entropy encoder
z1/z2/z3_bpp - average bpp at each layer
z2/z3LR-scl - average LR-scl at each layer
conloss - average loss of MISA
```
While an evaluation entry:
```
| eval  | S: 0 | ER: 6.3794 | z1_PSNR: 12.4813 | z2_PSNR: 12.4848 | z3_PSNR: 12.4852 | z1_bpp: 0.1356 | z2_bpp: 0.0721 | z3_bpp: 0.0256 | z2_LR-scl: 4.8209 | z3_LR-scl: 4.9656
```
which tells the expected reward and other metrics evaluating current policy after steps. Note that is average evaluation performance over episodes (usually 10):
```
eval - evaluating episode
S - total number of environment steps
ER - average evaluating episode reward
z1/z2/z3_PSNR - average PSNR at each layer
z1/z2/z3_bpp - average bpp at each layer
z2/z3_LR-scl - average LR-scl at each layer
```

## Train the HCL agent (i.e., continuous version)
To train an HCL agent on the task from image-based observations, navigate to the directory: [mlve/mlve-continuous/pytorch_sac_mlve_continuous/](https://github.com/dezhaochen/mlve/tree/main/mlve-continuous/pytorch_sac_mlve_continuous). The file contains the following command, which you can modify to try different environments / hyperparamters. ```source train.sh```
```
python train.py \
    --domain_name cartpole \
    --task_name swingup \
    --encoder_type pixel \
    --decoder_type pixel \
    --action_repeat 4 \
    --testpsnr \
    --work_dir ./logsave \
    --seed 1
```
The output in the console is similar to that in the discrete version.

# Train baselines
To train the other baselines on the task from image-based observations, navigate to the directory: [mlve/baseline/pytorch_sac_compression/](https://github.com/dezhaochen/mlve/tree/main/baseline/pytorch_sac_compression). The file contains the following command, which you can modify to try different environments / hyperparamters. ```source train.sh```

Compression methods could be seleted including JPEG, BPG and CompressAI through parameter ```--cp_method```. The default is to only use RL feedback for compression.