Github doesn't automatically render LaTex equations in markdown. See [readme.html](https://github.com/quevivasbien/math-finance/edit/master/readme.html) instead.

## Contract problems

The contract problems are principal-agent problems based on a model where an investor (principal) and a manager (agent) enter into a fixed-length contract where the investor pays the manager to manage a firm. The investor benefits from increasing firm value and pays the manager for his work. The catch is that the manager can appropriate some firm value for himself, reporting a different (lower) firm value to the investor. The problem is how the investor should make the payment contract to incentivize the manager not to appropriate too much.

The model is as follows:

The manager's continuation utility at a time $t$ (expected gain from continuing through the end of the contract) is
$$W_t = E_t \sum_{s=t+1}^T e^{-\rho (s-t)} \left( l_s + \theta a_s \right)$$
where $E_t$ is the expected value operator given information at time $t$, $T$ is the length of the contract, $\rho$ is the manager's rate (greater than the riskless rate), $l_t$ is the payment from the investor to the manager in period $t$, $a_t$ is the amount of firm value appropriated by the manager in period $t$, and $\theta \in [0, 1]$ represents the proportion of appropriated firm value that the manager can actually use (stealing is not totally efficient).

The most important manager's continuation utility to consider is $W_0$, which is the expected gain to the manager of entering into to the contract. We can assume that the manager will not accept any contract where $W_0 < W_{min}$, where $W_{min}$ is some nonnegative threshold.

The investor's continuation utility is
$$V_t = E_t \sum_{s=t+1}^T e^{-r(s-t)} \left( \Delta \hat{Y}_s - l_s \right)$$
Here $\hat{Y}_t$ is the reported firm value at time $s$. Specifically,
$$\hat{Y}_t = Y_t - \sum_{s=1}^t a_s,$$
where $Y_t$ is the true firm value, which we model as a jump-diffusion process
$$dY_t = Y_t (\mu dt + \sigma dZ_t + dJ_t)$$
($\mu$ is a growth rate parameter, $\sigma$ is a volatility parameter, $Z_t$ is a Brownian motion, and $J_t = \sum_{i=1}^{N_t} X_i$, where $N_t$ is a Poisson process with parameter $\lambda$ and the $X_i$s are N($m, s^2$) random variables.)

$\Delta \hat{Y}_t = \hat{Y}_t - \hat{Y}_{t-1} = Y_t - Y_{t-1} - a_t$, and $r$ is the riskless rate.

### Contracts with linear payments

If we assume that the investor chooses a payment scheme of the form
$$l_t = \alpha + \beta \hat{Y}_{t-1} + \delta \Delta \hat{Y}_{t}$$
then we can make some statements about the optimal strategies on the parts of the investor and the manager.

Since $\hat{Y}_t = Y_t - \sum_{s=1}^t a_s$, we can rewrite $l_t$ as
$$l_t = \alpha + \beta \left( Y_{t-1} - \sum_{s=1}^{t-1} a_s \right) + \delta \left( Y_t - Y_{t-1} - a_t \right)$$
The cost (in lost payment) to the manager of appropriating $a_t$ in time period $t$ is the negative of all the terms in the above expression that contain $a_t$ summed over the current and future time periods, which is
$$\delta a_t + \sum_{s=t+1}^T e^{\rho(s-t)} \beta a_t$$
The manager will only appropriate firm value if the cost is less than the benefit ($\theta a_t$), i.e., cancelling out $a_t$ (appropriate for $a_t > 0$) if
$$\delta + \sum_{s=t+1}^T e^{\rho(s-t)} \beta < \theta$$
If that condition is met, then the manager will appropriate as much as he can (if there are no restrictions in place, the manager will just rob the investor blind). It's realistic to assume that there is some maximum amount that the manager can appropriate in a given period; let's call that $a_{max}$. We also assume that $a_t$ cannot be negative. So we have
$$a_t = a_{max} 1 \left\{ \delta + \sum_{s=t+1}^T e^{\rho(s-t)} \beta < \theta \right\}$$
Having chosen $a_t$ for $t=1,\dots,T$, the manager will (as explained above) accept a contract where $W_0 \geq W_{min}$.

The problem of interest is how the investor can choose the parameters $\alpha$, $\beta$, and $\gamma$ so as to maximize $V_0$ given that the manager follows the optimal appropriation strategy and requires $W_0 \geq W_{min}$. The code in `contract_with_jumps.py` contains functions that calculate $W_0$ and $Y_0$, visualize $V_0$ for various payment strategies, and estimate the values of $\alpha$, $\beta$, and $\gamma$ that maximize $V_0$.

### Contracts with quadratic payments

If we assume that the investor chooses a payments scheme of the form
$$l_t = \alpha + \beta \hat{Y}_{t-1} + \gamma \hat{Y}_{t-1}^2 + \delta \Delta \hat{Y}_{t}$$
then the basic problem remains the same, but the manager's optimal appropriation strategy is considerably more complicated. Expanding $l_t$ as before, we get
$$\begin{align*}
l_t &= \alpha + \beta \left( Y_{t-1} - \sum_{s=1}^{t-1} a_s \right) + \gamma \left( Y_{t-1} - \sum_{s=1}^{t-1} a_s \right)^2 + \delta \left( Y_t - Y_{t-1} - a_t \right) \\
&= \alpha + \beta \left( Y_{t-1} - \sum_{s=1}^{t-1} a_s \right) + \gamma \left( Y_{t-1}^2 - 2 Y_{t-1} \sum_{s=1}^{t-1} a_s + \left[ \sum_{s=1}^{t-1} a_s \right]^2 \right) + \delta \left( Y_t - Y_{t-1} - a_t \right)
\end{align*}$$
so the cost of appropriating $a_t$ is
$$\delta a_t + \sum_{s=t+1}^T e^{-\rho(s-t)} \left( \gamma \left[ 2Y_{s-1} a_t - a_t^2 - 2 a_t \sum_{u=1}^{s-1} a_u 1 \{u \neq t\} \right] + \beta a_t \right)$$
so the manager should appropriate $a_t$ if (factoring out $a_t$)
$$\delta + \sum_{s=t+1}^T e^{-\rho(s-t)} \left( \gamma \left[ 2Y_{s-1} - a_t - 2 \sum_{u=1}^{s-1} a_u 1 \{u \neq t\} \right] + \beta \right) < \theta$$
You can see that the decision to appropriate $a_t$ depends both upon the current and future values of $Y_t$ and upon the *other* appropriation decisions, with the decision to appropriate at some other time making appropriation at the current time less costly.

The first step to resolving this decision rule is to recognize that the manager does not actually know $Y_{s-1}$ for $s-1 > t$; the manager has to rely on the *expected* cost of appropriation, so the actual inequality to be considered is
$$\delta + \sum_{s=t+1}^T e^{-\rho(s-t)} \left( \gamma \left[ 2 E_t Y_{s-1} - a_t - 2 \sum_{u=1}^{s-1} E_t a_u 1 \{u \neq t\} \right] + \beta \right) < \theta$$

Notice that it will always be desirable to appropriate if the manager is allowed to appropriate enough, so we again impose the restriction that $a_t \in [0, a_{max}]$. We have to treat $a_u$ as a random variable for $u > t$.

Also notice that the cost of appropriation will fall as $t$ increases, and will only rise if $Y_{s-1}$ turns out to be significantly larger than its expected value at time $t$. This suggests that we can make the assumption that if $a_t = a_{max}$ then $E_t a_u = a_{max}$ for every $u > t$. $E_t a_u$ is of course already known when $u \leq t$. The code in `contract_quadratic_payments.py` computes expected utilities for the manager and investor and determines the investor's optimal payment scheme ($\alpha$, $\beta$, $\gamma$, and $\delta$ coefficients), assuming that the manager appropriates following this strategy. However, this strategy may not actually be optimal in cases where $Y_t$ is very volatile (in which case $Y_{s-1}$ may frequently exceed its expected value as calculated in a previous time period). I'm still trying to figure out how serious this issue is. Determining an exactly optimal strategy seems almost hopelessly recursive, although there may be a way to do that.
