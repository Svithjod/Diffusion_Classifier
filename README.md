# Diffusion Classifier

This is a simple pytorch implementation of  [Your Diffusion Model is Secretly a Zero-Shot Classifier](http://arxiv.org/abs/2303.16203).

[model.py](model.py) is a minimal implementation of a conditional diffusion model with the ability of Bayesian Inference by Monte Carlo sampling. During training, it learns to generate MNIST digits conditioned on a class label. During inference, it samples pairs of $(t, \epsilon_{t})$ to estimate the bayes probability distribution of the given image. The neural network architecture is a small U-Net. This code is modified from [this excellent repo](https://github.com/TeaPearce/Conditional_Diffusion_MNIST).

The conditioning roughly follows the method described in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) (also used in [ImageGen](https://arxiv.org/abs/2205.11487)). The model infuses timestep embeddings $t_e$ and context embeddings $c_e$ with the U-Net activations at a certain layer $a_L$, via,

$$
a_{L+1} = c_e  a_L + t_e.
$$

At training time, $c_e$ is randomly set to zero with probability $0.1$, so the model learns to do unconditional generation (say $\psi(z_t)$ for noise $z_t$ at timestep $t$) and also conditional generation (say $\psi(z_t, c)$ for context $c$). This is important as at generation time, we choose a weight, $w \geq 0$, to guide the model to generate examples with the following equation,

$$
\hat{\epsilon}_{t} = (1+w)\psi(z_t, c) - w \psi(z_t).
$$

Increasing $w$ produces images that are more typical but less diverse.

The basic idea of a diffusion classifier is bayesian inference, that is, 

$$
p_\theta(\mathbf{c}_i\mid\mathbf{x})=\frac{p(\mathbf{c}_i)\ p_\theta(\mathbf{x}\mid\mathbf{c}_i)}{\sum_j p(\mathbf{c}_j)\ p_\theta(\mathbf{x}\mid\mathbf{c}_j))}
$$

A uniform prior over $\{\mathbf{c}_i\}$ (i.e., $p(\mathbf{c}_i) = \frac{1}{N}$) cancels all the $p(\mathbf{c})$ terms, and by further replacing $\text{log}\ p_\theta(\mathbf{x}\mid\mathbf{c})$ with ELBO, we have,
$$
\begin{aligned}
p_\theta(\mathbf{c}_i \mid \mathbf{x}) & \approx \frac{\exp \{-\mathbb{E}_{t, \epsilon}[\|\epsilon-\epsilon_\theta(\mathbf{x}_t, \mathbf{c}_i)\|^2]+C\}}{\sum_j \exp \{-\mathbb{E}_{t, \epsilon}[\|\epsilon-\epsilon_\theta(\mathbf{x}_t, \mathbf{c}_j)\|^2]+C\}} \\
& =\frac{\exp \{-\mathbb{E}_{t, \epsilon}[\|\epsilon-\epsilon_\theta(\mathbf{x}_t, \mathbf{c}_i)\|^2]\}}{\sum_j \exp \{-\mathbb{E}_{t, \epsilon}[\|\epsilon-\epsilon_\theta(\mathbf{x}_t, \mathbf{c}_j)\|^2]\}}
\end{aligned}
$$
which can be estimated by Monte Carlo sampling (see [Your Diffusion Model is Secretly a Zero-Shot Classifier](http://arxiv.org/abs/2303.16203)). 

During bayesian inference, we no longer drop $c_e$ for we estimate the relative confidence in every class given an image. And we set $w=0$, unlike the sampling procedure for generation. The model samples $N$ pairs of $(t, \epsilon_{t})$ for inference, the larger the $N$, the more accurate the estimation. 

We trained the conditioned diffusion model for **50** epochs, and performed sampling with $N=20,50,100$. The model was run on a single RTX 3090. The batch size was set to 512. 



| Sample | Acc   | Time    |
| ------ | ----- | ------- |
| 20     | 98.27 | ~14 min |
| 50     | 98.98 | ~35 min |
| 100    | 99.23 | ~70 min |

