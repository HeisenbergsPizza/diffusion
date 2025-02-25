# DiT

### Classifier-free guidance

回顾DDPM，训练优化目标是最大化似然函数 $p_{\theta}(\mathbf{x_0})$ 。而如何能使得模型生成符合我们希望的图片，则需要使用条件扩散模型，此时优化目标则变为 $p_\theta(\mathbf{x}_0|c)$，扩散过程的逆过程则变为 $p_\theta(\mathbf{x}_{t-1}|  \mathbf{x} _{t},c)$
