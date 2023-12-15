# DUQ: **D**istribution of **U**ncertainty **Q**uality
The idea is that for any prediction $f(x)$ made by a model, we expect the prediction to be drawn from a normal distribution with mean $\mu=\hat{f}$ and variance equal to the uncertainty prediction of the model $\sigma(x)^2$. Here $\hat{f}$ is the 'true' value. Which means:

$f(x) \sim \mathcal{N}(\hat{f},\sigma(x))$

$f(x) - \hat{f} \sim \mathcal{N}(0,\sigma(x))$

$\frac{f(x) - \hat{f}}{\sigma(x)} \sim \mathcal{N}(0,1) \qquad(1)$

Therefore, when we work out the residual divided by the uncertainty $(1)$ for each entry in our test data, $x_i$, we expect the resulting distribution to be normal with mean 0 and standard deviation 1. The DUQ is a way to compare the actual and theoretical distributions, thereby giving a measure of the quality of the model's uncertainty predictions. 

Smaller values indicate a smaller 'error' between the theoretical and actual distribution, meaning a better prediction of uncertainty. The DUQ's value is bounded between 0 and 1. 

A minimal code example for calculating the DUQ value using the true values, model predicitions, and model uncertainties calculated on test data:
```
from duq import DUQ

duq_calc = DUQ()
duq,_,_,_ = duq_calc(y_true,y_pred,y_unc)
```
Visualising the theoretical and actual distributions can also be useful. 
```
from duq import DUQ, plot_duq

duq_calc = DUQ()
duq,bins,ideal_hist,hist = duq_calc(y_true,y_pred,y_unc)
plot_duq(bins,ideal_hist,hist)
plt.show()
```

See this paper: https://doi.org/10.1016/j.commatsci.2021.110916 for further details of the theory behind the DUQ. 
