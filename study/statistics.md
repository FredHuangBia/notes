<h1 style="text-align: center;border-bottom: none;"> Statistics </h1>

# Confidence interval
## Central limit theorem
Given any distribution, if we independently draw multiple sets of samples, the distribution of the sample mean follow a normal distribution, even if the original distribution is not normal.

Specifically, if we have a population mean $\mu$ and a population standard deviation $\sigma$, and we sample $N$ instances from the population. The expected sample mean is $\mu$ and expected sample standard deviation is $\frac{\sigma}{\sqrt{N}}$. 

## Confidence interval
A confidence interval is a range of the estimation of an unknown value. Usually, the true value will be contained by the range with a probability. For example, if we sample from a normal distribution, with 95% of chance, the value will be within $\pm2\sigma$. Then $(\mu-2\sigma, \mu+2\sigma)$ will be the 95% confidence interval. 

A more useful example is to estimate the population mean from sample mean. Given any sample, the population mean will be within $(\mu-2SE, \mu+2SE)$ where $\mu$ and $SE=\frac{\sigma}{\sqrt{N}}$ are the sample mean and standard deviation.

## $z$-score
We could generalize the above sample mean confidence interval to
$$\mu \pm zSE$$
For 99%: $z=2.58$, for 95%: $z=1.96$, and for 90%: $z=1.65$.

## Estimating $\sigma$
The true population $\sigma$ is unknown, but we could estimate it with sample standard deviation. This is called boostrap.

# Hypothesis test
In hypothesis test, we usually have a null hypothesis which usually means "nothing significant is happening". We usually call it $H_0$. And on the other hand, we have an alternative hypothesis called $H_A$. We usually assume $H_0$ is true, and want our observations to provide us enough evidence to confidently reject the null hypothesis.

## $z$-statistic
A $z$-statistic (one type of test statistic), measures how far away the observed value is from our expected value under the assumption that the null hypothesis is ture.
$$z=\frac{observed-expected}{SE}$$
The $expected$ and $SE$ are all under the assumption that $H_0$ is true. The larger the $z$ value is, the stronger the evidence.

## $p$-value --- observed significance level
$p$ value is the probability of observing a more extreme $z$ under the assumption that $H_0$ is true. But it does not directly measure the probability of $H_0$. If $H_0$ is true, then $z$ follow standard normal distribution according to central limit theorem. Then the $p$ value is the probability of getting a $|z'|>|z|$, or $z'>z$, depending on whether we are doing one sided or two sided test. So if $p<0.05$ (we call this 5% significance level), then we say the observations is statistically significant and will reject the null hypothesis.

## Student's t-distribution
When the $\sigma$ is unknown, we estimate it from the sample standard deviation. However, when the sample size is small (usually <=20>), the estimation of $\sigma$ has a larger uncertainty. The normal curve in this case will not be a good enough approximation to the distribution of the z-statistic. Then it comes the student's t-distribution with $n-1$ degress of freedom. Now we estimate the population standard deviation by
$$s=\sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2}$$
In this case (when sample size is small), we usually replace the $z$ test with $t$ test.
$$\bar{x}\pm t_{n-1}SE$$
We replace the $z$ cut off with a $t$ cut off. For any sample size $N$, there's a student curve, the one we are going to use is the one with $N-1$ degrees of freedom.

## Errors
Type I error is when $H_0$ is true, but it get rejected.
Type II error is when $H_0$ is false, but it did not get rejected.

## Two sample z test
We will very often come into problems of whehter the two samples have the same mean. In this case, usually we assume the difference of the mean is 0, which is the $H_0$. Then we could use the following rule to find The $SE$ of the difference:
### Rule of standard errors
If two samples are drawn independently, then the standard error of their difference is
$$SE(v_1-v_2)=\sqrt{SE_1^2+SE_2^2}$$
 
# Resampling
## Monte Carlo Method
Sometimes, it is hard to estimate an actual value $\theta$ from the polulation, such as mean and std of a distribution. But we might be able to draw many random samples from the population, or **simulate** the prosee. We use the mean of these samples/simulations $\hat{\theta}$ to estimate the actual value. This is the Monte Carlo Method. 

The Monte Carlo estimated values $\hat{\theta}$ themselves are not oracle. So we could sample many times, and calculate the std of the sample statistics, to get the confidence interval of our Monte Carlo sampling.

## Bootstrap
Say we have an estimate $\hat{\theta}$ for a parameter $\theta$, and we want to know how accurate it is, i.e. we want to find $SE(\hat{\theta})$ and give $\hat{\theta}$ a confidence interval. However, in many situation, getting more than one sample is impossible. Then to estimate the std of the statistic from the random sample, we use bootstrap method.

First of all, we apply **plug-in** principle. We use the value computed from the sample to replace the oracle value. Let $\theta=\hat{\theta}$.

Then we draw many new samples, each new sample $X_1^*...X_n^*$ from our original sample $X_1...X_n$ with replacement. We use each new sample to calculate a value $\hat{\theta}^*$.

Finally we use these $\hat{\theta}^*$ to estimate the $SE(\hat{\theta})$.

We calculating confidence interval, we need to be careful that the $\hat{\theta}^*$ may not be a normal distribution. So in case it is not, we need to carefully picking the percentiles cutoffs.

## Bootstrap and regression
Bootstrap can also be used to estimate the confidence interval obtaiend by regression, such as least square. But instead of resample from the pairs directly, we resample from the residuals after the fit. And plut-in the estimated values (the fit) as the oracle, using the residuals, and assume the $X_i$ is fixed, we recalculate a $Y_i^*$. Now these new $X_i, Y_i^*$ pairs are new samples we draw, and we could use them to re-estimate the parameters (the fit).

# $\chi^2$-Tests
$\chi^2$-tests assumes samples are drawn independently.
![](imgs/chi-square%20distribution.png)
## Test of goodness of fit
We use M&M as an example, suppose we have a expected probability distribution of different color categories, and an observed color distribution. We assume $H_0$ that the observed fits the expectation. Now we calculate $\chi^2$ to find the p-value.
$$\chi^2=\sum_{categories}\frac{(observed-expected)^2}{expected}$$

The $\chi^2$ has its own distribution, the one we should pick is the one with (num_categories-1) degrees of freedom. The area to the right of the $\chi^2$ is the p-value. Larger $\chi^2$ or smaller p-value holds more evidence against $H_0$.

Notice that the $\chi^2$ test is a generalization of the z-test to multiple categories.

## Test of Homogeneity and Independence
Homogeneity tests assumes different categories are the same on some probability. It is for evaluating whether a categorical variable measured on **several samples** has the same distribution in each of the samples. In this case, if we don't know the probability, we could pool the samples and calculate the probability. Now we could run the above $\chi^2$ test. Note that the degrees of freedom is now different.
$$DoF=(num\;rows-1)\times (num\;cols-1)$$

Test of independence is similar to the case of homogeneity. There are two categorical variables, and there's only one sample.

|Test type     | Samples | Categorical Variables (diff from num categories) |
| ---          | ------- | ---                                              |
| homogeneity  | Many    | Single |
| independence | Single  | Two    |
