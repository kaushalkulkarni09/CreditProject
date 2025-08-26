### **Hybrid Model for Credit Risk Assessment and Derivative Pricing**

This project presents a comprehensive framework for modeling credit risk and pricing credit derivatives. It integrates a Nelson-Siegel-Svensson (NSS) model for interest rate dynamics with a stochastic CIR++ model for default intensity. The framework employs a rigorous Bayesian inference approach for parameter calibration, ensuring a statistically sound and transparent process. The entire project is implemented in Python, leveraging a robust set of open-source libraries.

**Yield Curve Modeling: The Nelson-Siegel-Svensson (NSS) Model**

The project begins by calibrating the U.S. Treasury par yield curve, a fundamental component for any financial valuation. The NSS model is chosen for its ability to capture a wide range of yield curve shapes (including upward-sloping, downward-sloping, and humped) with a small, parsimonious set of parameters.

**Data Source & Calibration**
The model was fitted to daily U.S. Treasury Par Yields from 2025-05-28 to 2025-08-22, fetched from the Federal Reserve Economic Data (FRED) API. The calibrated parameters for the most recent trading day, 2025-08-26, are as follows:

| Parameter | Value |
| :--- | :--- |
| $\beta_0$ | $0.044124$ |
| $\beta_1$ | $-0.005343$ |
| $\beta_2$ | $-0.026508$ |
| $\beta_3$ | $0.007658$ |
| $\lambda_1$ | $0.349299$ |
| $\lambda_2$ | $10.000000$ |

These parameters successfully capture the upward-sloping yield curve observed in the market data. This calibrated curve is then used to derive key metrics for subsequent valuations.

| Metric | NSS Value |
| :--- | :--- |
| Discount Factor (1-Year) | $0.9651$ |
| Discount Factor (5-Year) | $0.8054$ |
| Zero Rate (1-Year) | $3.5506\%$ |
| Zero Rate (5-Year) | $4.3281\%$ |

**Synthetic CDS Market Data Generation**

For testing and demonstration, a term structure of Credit Default Swap (CDS) spreads was synthetically generated. This data simulates the observable market prices for credit risk across various maturities, providing the necessary input for model calibration.

| Maturity (Years) | Spread (bps) |
| :--- | :--- |
| $1.0$ | $119.92$ |
| $2.0$ | $129.01$ |
| $3.0$ | $138.43$ |
| $5.0$ | $146.16$ |
| $7.0$ | $153.20$ |
| $10.0$ | $165.55$ |

**CIR++ Default Intensity Model Calibration: Bayesian Inference**

The core of this project is the calibration of the CIR++ stochastic default intensity model to the simulated CDS spreads. This model describes a credit default intensity that follows a mean-reverting stochastic process.

The model is calibrated using Bayesian inference with a Markov Chain Monte Carlo (MCMC) sampler. This approach provides a full posterior distribution for the model parameters, capturing the uncertainty inherent in the calibration process. The MCMC sampler was run with 2 chains, each with 2000 tune-in steps (warmup) and 5000 draws for the final posterior distribution.

**Posterior Summary Statistics**

The following table presents the key statistics from the posterior distributions of the calibrated parameters:

| Parameter | Mean | Std. Dev. | 94% HDI Range |
| :--- | :--- | :--- | :--- |
| $\kappa_{\lambda}$ (Mean Reversion Speed) | $0.518$ | $0.047$ | $[0.435, 0.612]$ |
| $\sigma_{\lambda}$ (Volatility) | $0.187$ | $0.019$ | $[0.155, 0.222]$ |
| $\sigma_{obs}$ (Observation Error) | $0.00052$ | $0.00007$ | $[0.00040, 0.00065]$ |

The low standard deviations and narrow High-Density Intervals (HDI) indicate that the model parameters are well-identified by the market data.

**Monte Carlo CDS Pricing**

With the calibrated CIR++ model, a Monte Carlo simulation is used to price a CDS. The simulation jointly models the stochastic dynamics of the default intensity and the risk-free rate, allowing for a robust valuation that accounts for the correlation between these two factors. The CDS price is calculated as the present value of the expected loss from default minus the present value of the expected premium payments.

| CDS Input | Value |
| :--- | :--- |
| Notional | $\$1,000,000$ |
| Maturity | $5.0$ years |
| Coupon | $1.50\%$ (paid quarterly) |

The Monte Carlo simulation yielded the following price:

**Final Monte Carlo CDS Price:** $5,602.85

**Risky Bond Valuation**

Finally, the framework is used to price a risky bond, incorporating both the time value of money from the NSS yield curve and the credit risk from the calibrated CIR++ model. The valuation is performed by calculating the present value of expected future cash flows, adjusted by the probability of survival.

| Bond Input | Value |
| :--- | :--- |
| Face Value | $\$1,000$ |
| Coupon Rate | $3.00\%$ |
| Maturity | $7.0$ years |

**Estimated Risky Bond Price:** $955.42

This price is below the face value, reflecting the credit risk embedded in the bond, and is consistent with the simulated market data.

**Reproducibility**

For the purpose of reproducibility, this project relies on a consistent software environment and a fixed random number seed. The core libraries used are:

* **Python**
* **Numpy**
* **Pandas**
* **SciPy**
* **PyMC**
* **Requests**
* **Matplotlib**
* **Seaborn**

To reproduce the results, ensure you have the correct software environment, and then execute the main script from your terminal: **python creditproject.py**




