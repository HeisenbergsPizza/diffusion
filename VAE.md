# VAE

主要参考博客：https://lilianweng.github.io/posts/2018-08-12-vae/

## Autoencoder

![1740109300281](image/VAE/1740109300281.png)

Encoder本质上完成了对数据的降维

encoder function $g_\phi(.)$

decoder function $f_\theta(.)$

$x'=f_\theta(g_\phi(x))$

$(\theta,\phi )$ 由通过学习以重建数据样本（可通过多种评价指标训练），本质上学习的是一个恒等函数

## Variational Autoencoder

相比于将输入映射为固定向量，我们可以将其映射为一个分布。假设这个分布为 $p_\theta$ ，参数化为 $\theta$ 。则数据输入 $\mathbf{x}$ 和 latent encoding vector $\mathbf{z}$ 可以被定义为：

* 先验 $p_\theta(\mathbf{z})$ （假设服从标准正态分布）
* 似然 $p_\theta(\mathbf{x}|\mathbf{z})$
* 后验 $p_\theta(\mathbf{z}|\mathbf{x})$

取样 $\mathbf{z}^{(i)}$ 来自先验分布 $p_\theta(\mathbf{z})$ ，然后由条件分布 $p_\theta(\mathbf{x}|\mathbf{z}=\mathbf{z}^{(i)})$ 生成 $\mathbf{x}^{(i)}$，注意这里的条件分布是设定好的生成模型。

则最优参数 $\theta^{*}$ 满足 $\theta^{*}=\arg \underset{\theta}{\max} \prod_{i=1}^{n} p_\theta(\mathbf{x}^{(i)})$

<=>   $\theta^{*}=\arg \underset{\theta}{\max} \sum_{i=1}^{n} \log p_\theta(\mathbf{x}^{(i)})$

其中   $p_\theta(\mathbf{x}^{(i)})=\int p_\theta(\mathbf{x}^{(i)}|\mathbf{z}) p_\theta(\mathbf{z})d\mathbf{z}$

但是这种形式想要计算需要对 $p_\theta(\mathbf{x}^{(i)}|\mathbf{z}) $ 使用贝叶斯公式，就会需要后验分布 $p_\theta(\mathbf{z}|\mathbf{x}^{(i)})$ 的表达式，而这个表达式不可直接获取，因此需要一种新的近似函数 $q_\phi(\mathbf{z}|\mathbf{x})$，参数化为 $\phi$

![1740126585962](image/VAE/1740126585962.png)

这里可以把  $\mathbf{z} \overset{p_\theta(\mathbf{x}|\mathbf{z})}{\rightarrow} \mathbf{x}$ 类比Autoencoder的decode过程

把 $\mathbf{x} \overset{q_\phi(\mathbf{z}|\mathbf{x})}{\rightarrow} \mathbf{z}$ 类比为Autoencoder的encode过程

总结：$p_\theta( \mathbf{x}|\mathbf{z})$ 是一个定义好的解码器，在训练过程中想要得到最优的 $\theta$，需要优化的目标中含有后验分布 $p_\theta(\mathbf{z}|\mathbf{x}^{(i)})$ 难以获取表达式，因此我们再引入一个 $q_\phi(\mathbf{z}|\mathbf{x})$ 近似 $p_\theta(\mathbf{z}|\mathbf{x}^{(i)})$

### 损失函数：ELBO

我们可以使用**Kullback–Leibler divergence**来量化两个分布之间的距离，$D_{KL}(\mathbf{X}|\mathbf{Y})$ 测量如果使用分布 $\mathbf{Y}$ 来表示分布 $\mathbf{X}$ 会损失多少信息。

$D_{KL}(P \| Q)=\sum_{x\in \chi}P(x) \log(\frac{P(x)}{Q(x)})$

因此我们希望最小化 $D_{KL}(q_\phi(\mathbf{z}|\mathbf{x})| p_\theta(\mathbf{z}|\mathbf{x}) )$ 关于 $\phi$

![1740146750001](image/VAE/1740146750001.png)

$$
\begin{aligned}
& D_\text{KL}( q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z}\vert\mathbf{x}) ) & \\
&=\int q_\phi(\mathbf{z} \vert \mathbf{x}) \log\frac{q_\phi(\mathbf{z} \vert \mathbf{x})}{p_\theta(\mathbf{z} \vert \mathbf{x})} d\mathbf{z} & \\
&=\int q_\phi(\mathbf{z} \vert \mathbf{x}) \log\frac{q_\phi(\mathbf{z} \vert \mathbf{x})p_\theta(\mathbf{x})}{p_\theta(\mathbf{z}, \mathbf{x})} d\mathbf{z} & \scriptstyle{\text{; Because }p(z \vert x) = p(z, x) / p(x)} \\
&=\int q_\phi(\mathbf{z} \vert \mathbf{x}) \big( \log p_\theta(\mathbf{x}) + \log\frac{q_\phi(\mathbf{z} \vert \mathbf{x})}{p_\theta(\mathbf{z}, \mathbf{x})} \big) d\mathbf{z} & \\
&=\log p_\theta(\mathbf{x}) + \int q_\phi(\mathbf{z} \vert \mathbf{x})\log\frac{q_\phi(\mathbf{z} \vert \mathbf{x})}{p_\theta(\mathbf{z}, \mathbf{x})} d\mathbf{z} & \scriptstyle{\text{; Because }\int q(z \vert x) dz = 1}\\
&=\log p_\theta(\mathbf{x}) + \int q_\phi(\mathbf{z} \vert \mathbf{x})\log\frac{q_\phi(\mathbf{z} \vert \mathbf{x})}{p_\theta(\mathbf{x}\vert\mathbf{z})p_\theta(\mathbf{z})} d\mathbf{z} & \scriptstyle{\text{; Because }p(z, x) = p(x \vert z) p(z)} \\
&=\log p_\theta(\mathbf{x}) + \mathbb{E}_{\mathbf{z}\sim q_\phi(\mathbf{z} \vert \mathbf{x})}[\log \frac{q_\phi(\mathbf{z} \vert \mathbf{x})}{p_\theta(\mathbf{z})} - \log p_\theta(\mathbf{x} \vert \mathbf{z})] &\\
&=\log p_\theta(\mathbf{x}) + D_\text{KL}(q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z})) - \mathbb{E}_{\mathbf{z}\sim q_\phi(\mathbf{z}\vert\mathbf{x})}\log p_\theta(\mathbf{x}\vert\mathbf{z}) &
\end{aligned}
$$

所以有：$\log p_\theta(\mathbf{x}) - D_\text{KL}( q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z}\vert\mathbf{x}) ) = \mathbb{E}_{\mathbf{z}\sim q_\phi(\mathbf{z}\vert\mathbf{x})}\log p_\theta(\mathbf{x}\vert\mathbf{z}) - D_\text{KL}(q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z}))$

左式中的对数似然 $\log p_\theta(\mathbf{x})$ 正是我们希望最大化的，而我们同时希望最小化实际后验分布与估计后验分布（此时KL散度类似于正则项），由于KL散度的非负性，我们可以定义我们的损失函数如下：

$$
\begin{aligned}
L_\text{VAE}(\theta, \phi) 
&= -\log p_\theta(\mathbf{x}) + D_\text{KL}( q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z}\vert\mathbf{x}) )\\
&= - \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z}\vert\mathbf{x})} \log p_\theta(\mathbf{x}\vert\mathbf{z}) + D_\text{KL}( q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z}) ) \\
\theta^{*}, \phi^{*} &= \arg\min_{\theta, \phi} L_\text{VAE}
\end{aligned}
$$

该损失函数即为变分下界，通过最小化损失，即可最大化生成真实数据样本的概率的下限。

https://kexue.fm/archives/5343 这篇博客中提到了使用联合分布可以更加直接地推得损失函数。

### Reparameterization Trick

损失函数中的期望项调用 $z\sim q_\phi(\mathbf{z}|\mathbf{x})$，由于此时 $\phi$ 对 $z$ 来说不是一个显式可微的函数，也就是说要把这个随机的采样改写为确定性函数与一个带系数的标准正态分布和。

![1740209649786](image/VAE/1740209649786.png)

本质上是将原先由依赖参数 $\phi$ 的随机变量 $\mathbf{z}$ 拆分成了一个关于 $\phi$ 的确定函数和一个与 $\phi$ 无关的随机噪声，将随机性从参数中剥离出来，使得 $\mathbf{z}$ 对参数可微，进而使得参数可以被训练。
