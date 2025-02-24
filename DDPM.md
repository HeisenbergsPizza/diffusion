## DDPM

![1739713593904](image/0214/1739713593904.png)

### Forward diffusion process

向原始图片上逐步加高斯噪声，分T次，该过程可视为马尔可夫过程，数学形式如下：

$q(x_{t}|x_{t-1} )=N(x_{t};\sqrt{1-\beta _{t}}x_{t-1},\beta_{t}\mathbf{I})$

其中$\beta_{t}$是高斯分布方差的超参数, $\beta_{1}<\beta_{2}<...<\beta_{T}$

超参数如何设置？  --线性插值

$q(x_{1:T}|x_{0})=\prod_{t=1}^{T}q(x_{t}|x_{t-1})=\prod_{t=1}^{T}N(x_{t};\sqrt{1-\beta _{t}}x_{t-1},\beta_{t}\mathbf{I})$

令 $\alpha_{t}=1-\beta_{t}$，且 $\overline{\alpha_{t}}=\prod_{i=1}^{t}\alpha_{i}$

$x_{t}=\sqrt{\alpha_{t}}x_{t-1}+\sqrt{1-\alpha_{t}}\epsilon_{1}$，其中 $\epsilon_{1}\sim N(0,\mathbf{I})$ 服从标准正态分布

迭代得 $x_{t}=\sqrt{\alpha_{t}}(\sqrt{\alpha_{t-1}}x_{t-2}+\sqrt{1-\alpha_{t-1}}\epsilon_{2})+\sqrt{1-\alpha_{t}}\epsilon_{1}$

由于$\epsilon_{1}$,$\epsilon_{2}\sim N(0,\mathbf{I})$，根据正态分布的性质: $N(\mu_{1},\sigma_1^2)+N(\mu_{2},\sigma_2^2)=N(\mu_{1}+\mu_{2},\sigma_1^2+\sigma_2^2)$

可得：$x_{t}=\sqrt{\alpha_{t}\alpha_{t-1}}x_{t-2}+\sqrt{1-\alpha_{t}\alpha_{t-1}}\overline{\epsilon_{2}}$， 其中 $\overline{\epsilon_{2}}\sim N(0,\mathbf{I})$

递推可得：$x_{t}=\sqrt{\overline{\alpha_{t}}} x_{0}+\sqrt{1-\overline{\alpha_{t}}} \overline{\epsilon_{t}}$， 其中 $\overline{\epsilon_{t}}\sim N(0,\mathbf{I})$

因此，任意时刻 $x_{t}$ 满足 $q(x_{t}|x_{0})=N(x_{t};\sqrt{\overline{\alpha_{t}}}x_{0},(1-\overline{\alpha_{t}})\mathbf{I})$

### Reverse diffusion process

目标是反向推导以上过程，从 $x_{T}$ 中逐步去除噪声，还原 $x_{0}$，就能实现数据生成。这需要对逆条件概率分布 $q(x_{t-1}|x_{t})$ 进行采样。但是这涉及整个数据集，难以直接计算，因此需要学习一个近似的模型

$$
p_{\theta}(x_{t-1}|x_{t})=N(x_{t-1};\mu_{\theta}(x_{t},t),\Sigma_{\theta}(x_{t},t))
$$

* 均值 $\mu_{\theta}(x_{t},t)$ 由神经网络预测
* 方差 $\Sigma_{\theta}(x_{t},t)$ 可以学习也可以设定为固定值

整个反向扩散过程的联合概率：$p_{\theta }(x_{0:T})=p(x_{T})\prod_{t=1}^{T}p_{\theta }(x_{t-1}|x_{t})$

扩散模型的逆过程由条件概率分布表示：

$$
q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\boldsymbol{\mu}}(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t \mathbf{I})
$$

利用贝叶斯定理，我们可以展开：

$$
q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) = \frac{q(\mathbf{x}_t | \mathbf{x}_{t-1}, \mathbf{x}_0) q(\mathbf{x}_{t-1} | \mathbf{x}_0)}{q(\mathbf{x}_t | \mathbf{x}_0)}
$$

由高斯分布的概率密度函数：$p(x)=\frac{1}{\sqrt{2\pi\sigma^2}} exp(-\frac{(x-\mu)^2}{2\sigma^2})$ 可推得：方差 $\tilde{\beta}_t = \frac{1 - \overline{\alpha}_{t-1}}{1 - \overline{\alpha}_t} \beta_t$ 是定值，而均值

$$
\tilde{\boldsymbol{\mu}}(\mathbf{x}_t, \mathbf{x}_0) = \frac{\sqrt{\alpha_t}(1 - \overline{\alpha}_{t-1})}{1 - \overline{\alpha}_t} \mathbf{x}_t + \frac{\sqrt{\overline{\alpha}_{t-1}} \beta_t}{1 - \overline{\alpha}_t} \mathbf{x}_0
$$

是一个依赖 $x_0$ 和 $ x_t$ 的函数

在去噪过程中，我们可以用噪声预测：

$$
\mathbf{x}_0 \approx \frac{1}{\sqrt{\overline{\alpha}_t}} (\mathbf{x}_t - \sqrt{1 - \overline{\alpha}_t} \boldsymbol{\epsilon}_t)
$$

最终去噪均值可以写成：

$$
\tilde{\boldsymbol{\mu}}_t = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \overline{\alpha}_t}} \boldsymbol{\epsilon}_t \right)
$$

现在要在已知 $x_{0}$（训练时）的情况下最大化对数似然，等价于最小化负对数似然 $-log p_{\theta}(x_{0})$

可以使用变分下界来最小化负对数似然：

$$
\begin{align*}
    -\log p_{\theta}(\mathbf{x}_0) &\leq -\log p_{\theta}(\mathbf{x}_0) + D_{\mathrm{KL}}(q(\mathbf{x}_{1:T}|\mathbf{x}_0) || p_{\theta}(\mathbf{x}_{1:T}|\mathbf{x}_0))    ；--KL散度非负 \\
    &= -\log p_{\theta}(\mathbf{x}_0) + \mathbb{E}_{\mathbf{x}_{1:T} \sim q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[ \log \frac{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}{p_{\theta}(\mathbf{x}_{1:T}) / p_{\theta}(\mathbf{x}_0)} \right] \\
    &= -\log p_{\theta}(\mathbf{x}_0) + \mathbb{E}_q \left[ \log \frac{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}{p_{\theta}(\mathbf{x}_{1:T})} \right] + \log p_{\theta}(\mathbf{x}_0) \\
    &= \mathbb{E}_q \left[ \log \frac{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}{p_{\theta}(\mathbf{x}_{1:T})} \right]
\end{align*}
$$

$$
L_{\mathrm{VLB}} = \mathbb{E}_{q(\mathbf{x}_{1:T})} \left[ \log \frac{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}{p_{\theta}(\mathbf{x}_{1:T})} \right] \geq -\mathbb{E}_{q(\mathbf{x}_0)} \log p_{\theta}(\mathbf{x}_0)
$$

利用 Jensen 的不等式也可以得到同样的结果。假设我们把交叉熵最小化作为目标

$$
\begin{align*}
    L_{\mathrm{CE}} &= -\mathbb{E}_{q(\mathbf{x}_0)} \log p_{\theta}(\mathbf{x}_0) \\
    &= -\mathbb{E}_{q(\mathbf{x}_0)} \log \left( \int p_{\theta}(\mathbf{x}_{0:T}) d\mathbf{x}_{1:T} \right) \\
    &= -\mathbb{E}_{q(\mathbf{x}_0)} \log \left( \int q(\mathbf{x}_{1:T}|\mathbf{x}_0) \frac{p_{\theta}(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} d\mathbf{x}_{1:T} \right) \\
    &= -\mathbb{E}_{q(\mathbf{x}_0)} \log \left( \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \frac{p_{\theta}(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \right) \\
    &\leq -\mathbb{E}_{q(\mathbf{x}_{0:T})} \log \frac{p_{\theta}(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \\
    &= \mathbb{E}_{q(\mathbf{x}_{0:T})} \left[ \log \frac{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}{p_{\theta}(\mathbf{x}_{0:T})} \right] = L_{\mathrm{VLB}}
\end{align*}
$$

为了将方程中的每个项转换为可分析计算的，目标可以进一步重写为几个 KL 散度和熵项的组合

$$
\begin{align*}
    L_{\mathrm{VLB}} &= \mathbb{E}_{q(\mathbf{x}_{0:T})} \left[ \log \frac{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}{p_{\theta}(\mathbf{x}_{0:T})} \right] \\
    &= \mathbb{E}_q \left[ \log \frac{\prod_{t=1}^{T} q(\mathbf{x}_t|\mathbf{x}_{t-1})}{p_{\theta}(\mathbf{x}_{0:T}) \prod_{t=1}^{T} p_{\theta}(\mathbf{x}_t|\mathbf{x}_{t-1})} \right] \\
    &= \mathbb{E}_q \left[ -\log p_{\theta}(\mathbf{x}_T) + \sum_{t=1}^{T} \log \frac{q(\mathbf{x}_t|\mathbf{x}_{t-1})}{p_{\theta}(\mathbf{x}_t|\mathbf{x}_{t-1})} \right] \\
    &= \mathbb{E}_q \left[ -\log p_{\theta}(\mathbf{x}_T) + \sum_{t=2}^{T} \log \frac{q(\mathbf{x}_t|\mathbf{x}_{t-1})}{p_{\theta}(\mathbf{x}_t|\mathbf{x}_{t-1})} + \log \frac{q(\mathbf{x}_1|\mathbf{x}_0)}{p_{\theta}(\mathbf{x}_1|\mathbf{x}_0)} \right] \\
    &= \mathbb{E}_q \left[ -\log p_{\theta}(\mathbf{x}_T) + \sum_{t=2}^{T} \log \frac{q(\mathbf{x}_t|\mathbf{x}_{t-1}, \mathbf{x}_0)}{p_{\theta}(\mathbf{x}_t|\mathbf{x}_{t-1})} + \log \frac{q(\mathbf{x}_1|\mathbf{x}_0)}{p_{\theta}(\mathbf{x}_1|\mathbf{x}_0)} \right] \\
    &= \mathbb{E}_q \left[ -\log p_{\theta}(\mathbf{x}_T) + \sum_{t=2}^{T} D_{\mathrm{KL}}(q(\mathbf{x}_t|\mathbf{x}_{t-1}, \mathbf{x}_0) || p_{\theta}(\mathbf{x}_t|\mathbf{x}_{t-1})) - \log p_{\theta}(\mathbf{x}_0|\mathbf{x}_1) \right] \\
    &= \mathbb{E}_q \left[ D_{\mathrm{KL}}(q(\mathbf{x}_T|\mathbf{x}_0) || p_{\theta}(\mathbf{x}_T)) + \sum_{t=2}^{T} D_{\mathrm{KL}}(q(\mathbf{x}_t|\mathbf{x}_{t-1}, \mathbf{x}_0) || p_{\theta}(\mathbf{x}_t|\mathbf{x}_{t-1})) - \log p_{\theta}(\mathbf{x}_0|\mathbf{x}_1) \right]
\end{align*}
$$

* $L_{T}=D_{\mathrm{KL}}(q(\mathbf{x}_T|\mathbf{x}_0) || p_{\theta}(\mathbf{x}_T))$
* $L_{t-1}=\sum_{t=2}^{T} D_{\mathrm{KL}}(q(\mathbf{x}_t|\mathbf{x}_{t-1}, \mathbf{x}_0) || p_{\theta}(\mathbf{x}_t|\mathbf{x}_{t-1}))$
* $L_{0}=-log p_{\theta}(\mathbf{x}_0|\mathbf{x}_1)$
* $L_{LVB}=L_{T}+L_{T-1}+···+L_{0}$

在 $L_{\mathrm{VLB}}$ 中，除了 $L_{0}$ 以外的所有 KL 散度项都在比较两个高斯分布，因此它们可以被闭式计算（有解析解）。由于 $\mathbf{x}_T$ 只是高斯噪声且并不包含可学习的参数，所以 $L_T$ 是一个常数，可以在训练过程中忽略。Ho 等人在 2020 年的工作中，通过一个独立的离散解码器（从 $x_0 \sim p_\theta(x_0|x_1)$ ,$x_1 \sim p_\theta(x_1|x_2)$…,$x_T \sim p_\theta(x_T)$ 过程推导而来）来对 $L_0$ 进行建模。

#### 对 $L_t$ 进行重参数化

原先的 $L_t$

现在需要神经网络来估计反向扩散中的条件概率分布，$p_{\theta}(x_{t-1}|x_t)= \mathcal{N}\bigl(x_{t-1};\,\mu_{\theta}(x_t,t),\,\Sigma_{\theta}(x_t,t)\bigr)$

想要训练出来 $\mu_{\theta}(x_t,t)$ 以预测从 $x_t$ 到 $x_{t-1}$ 的“去噪”均值，或者说让网络学会如何从带噪声的 $x_t$ 中恢复原始数据 $x_0$（或等价地预测每一步的噪声 $\epsilon$）

在实际实现中，我们常常假设 $\Sigma_{\theta}(x_t,t)$ 为常数或固定不变，并让神经网络直接预测输入 $x_t$ 中所包含的噪声 $\epsilon$。常见的做法是将网络输出表示为：

$$
\mu_{\theta}(x_t,t)
= \frac{1}{\sqrt{\alpha_t}}\Bigl(x_t - \frac{1 - \alpha_t}{\sqrt{1 - \overline{\alpha}_t}}\,\epsilon_{\theta}(x_t,t)\Bigr)
$$

其中 $\{\alpha_t\}$（以及 $\{\overline{\alpha}_t\}$）是扩散过程中的超参数序列。这样一来，在逆向过程中，我们对 $x_{t-1}$ 的采样可以写为：

$$
x_{t-1} \sim \mathcal{N}\bigl(x_{t-1};\,\mu_{\theta}(x_t,t),\,\Sigma_{\theta}(x_t,t)\bigr)
$$

给定损失项被重参数化为最小化与 $\mu$ 的差异：

$$
L_t 
= \mathbb{E}_{x_0,\;\epsilon \sim \mathcal{N}(0,I)} 
\Bigl[
\bigl\|
\epsilon 
- 
\epsilon_\theta\bigl(\sqrt{\alpha_t}\,x_0 + \sqrt{1 - \alpha_t}\,\epsilon,\; t\bigr)
\bigr\|^2
\Bigr].
$$

**Simplification**

Ho 等人（2020）发现，训练扩散模型时，如果忽略加权项，使用一个简化后的目标函数效果更好：

$$
\tilde{L}_t
= 
\mathbb{E}_{x_0,\;\epsilon \sim \mathcal{N}(0,I)}
\Bigl[
\bigl\|
\epsilon 
- 
\epsilon_\theta(x_t,\; t)
\bigr\|^2
\Bigr].
$$

最终的单一目标函数是：$L_{\mathrm{simple}}=\sum_{t=1}^T\tilde{L}_t+C$

![1739781589123](image/0214/1739781589123.png)
