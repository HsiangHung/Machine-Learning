
# Z-Test

We use Z-test when:
* For comparing sample means with population, and we **know** population standard deviation.
* For comparing sample proportion with population proportion.

Otherwise, we may need to use [t-test](https://github.com/HsiangHung/Machine_Learning_Note/tree/master/Statistics/hypothesis_test/T-test).

## Comparing Means: Numeric Variable

Note in the following section, we assume variables are continuous numerics  (e.g. average cost is $12 etc) and the population standard deviations are known.

### One Sample Z-Test: One Tailed

[[Statistics How to]][Hypothesis Testing] A principal at a certain school claims that the students in his school are above average intelligence. A random sample of thirty students IQ scores have a mean score of 112.5. Is there sufficient evidence to support the principal’s claim? The mean population IQ is 100 with a standard deviation of 15.

* State the null hypothesis: the students have average IQ as the population mean, so: $H_0$: $\mu = 100$.
* State the alternate hypothesis: the students of the school have above average IQ scores, so: $H_a$: $\mu > 100$.
* State the alpha level. Use default 5%. For one sample Z-test and 95%, the critical Z-score $Z_{c} = 1.65$.
* Calculate the test statistic

$$Z = \frac{\bar{x}-\mu_0}{\sigma/\sqrt{n}} = \frac{112.5-100}{15/\sqrt{30}}=4.56.$$

* $Z = 4.56 > Z_c=1.65$, so the p-value $<0.05$.  Therefore we reject the null hypothesis; i.e. the students' IQ are higher than mean population IQ.


### One Sample Z-Test: Two Tailed

[[Statistics How to]][Hypothesis Testing] Blood glucose levels for obese patients have a mean of 100 with a standard deviation of 15. A researcher thinks that a diet high in raw cornstarch will have a positive or negative effect on blood glucose levels. A sample of 30 patients who have tried the raw cornstarch diet have a mean glucose level of 140. Test the hypothesis that the raw cornstarch had an effect.

* State the null hypothesis: $H_0$: $\mu=100$.
* State the alternate hypothesis: $H_a$: $\mu \ne 100$.
* State significant level. We’ll use $\alpha = 0.05$ for this example. As this is a two-tailed test, split the alpha into two: 0.05/2=0.025. The critical Z-score for 97.5% is  $Z_c = 1.96$.
* Calculate test statistic

$$ Z = \frac{140 – 100}{15/\sqrt{30}} = 14.60.$$

* Since 14.6 > $Z_c$, we reject the null hypothesis.

## Categorical Variables

Note in the following section, we assume the variables are categorical (e.g. booking rate is 85% etc), and sampling obeys the binomial distribution.

### CLT for Proportion 

From central limit theorem, we have

$$\hat{p} \sim N \Big(\textrm{mean}=p_0, \textrm{SE}=\sqrt{\frac{p_0(1-p_0)}{n}} \Big),$$

for single proportion. Assume $\hat{p}$ is the sample proportion, like number of clicks divided by number of lands, and $p_0$ is the population proportion. The sample size is still $n$.

To estimate a proportion, the confidenece interval is given by 

$$ \textrm{point estimate} \pm \textrm{margin of error} = \hat{p} \pm z^* \textrm{SE}_{\hat{p}},$$

where

$$\textrm{SE}_{\hat{p}}=\sqrt{\frac{\hat{p}(1-\hat{p})}{n}}.$$

The **success-failure** condition for assuming single proportion on nearly normal distribution is at least 10 successes and 10 failures in the sample. Thus 

$$ np \ge 10$$ and $$n(1-p) ≥ 10.$$


### Single Proportion Z-Test

For comparing a sample proportion with the population proportion. The null and alternative hypotheses are separately

$$\textrm{H}_0: \hat{p} = p_0, \ \textrm{H}_a: \hat{p} \ne p_0.$$

The test statistic is z-test, defined as [[Stattrek: Hypothesis Test for a Proportion]][Stattrek, Hypothesis Test for a Proportion]

$$z = \frac{\hat{p}-p_0}{\sqrt{p_0(1-p_0)/n}}.$$

#### Example 1

90% of all plants species are classified as angiosperms (flowering
plants). If you were to randomly sample 200 plants from the list of
all known plant species, what is the probability that at least 95% of
plants in your sample will be flowering plants?

$$ \hat{p} \sim N \Big( \textrm{mean}=0.9, \textrm{SE}=\sqrt{\frac{0.9*0.1}{200}}\Big),$$

then z-score $Z = (0.95-0.9)/0.0212 = 2.36,$ and then the probability $P(\hat{p} > 0.95, Z > 2.36) = 0.0091$.

#### Example 2

A 2013 Pew Research poll found that 60% of 1,983 randomly sampled American adults believe in evolution. Does this provide convincing evidence that majority of Americans believe in evolution? "majority" means larger than 50%.

$$\textrm{H}_0: \hat{p} = p_0 = 0.5, \ \textrm{H}_a: p > 0.5.$$

In the problem, we have $\hat{p} = 0.6, n = 1983$. Thus the test-statistics is 

$$Z = \frac{0.6-0.5}{\sqrt{\frac{0.5\times 0.5}{1983}}} = 8.92.$$

Since p-value for $P(Z > 8.92)$ is very small, we can reject $H_0$.


### Difference Between Two Proportions

To estimate the difference between two proportions, the confidenece interval is given by (by [Coursera Inferential Statistics](https://www.coursera.org/learn/inferential-statistics-intro/lecture/kI4Ma/estimating-the-difference-between-two-proportions))

$$ (\hat{p}_1 - \hat{p}_2 ) \pm z^* \textrm{SE}_{\hat{p}_1 - \hat{p}_2},$$

where 

$$\textrm{SE}_{\hat{p}_1 - \hat{p}_2}=\sqrt{\frac{\hat{p}_1(1-\hat{p}_1)}{n_1} + \frac{\hat{p}_2(1-\hat{p}_2)}{n_2}}.$$

Each sample needs to meet **success-failure** condition:
* $n_1 p_1 \ge 10$ and $n_1(1-p_1) ≥ 10$. 
* $n_2 p_2 \ge 10$ and $n_2(1-p_2) ≥ 10$. 

#### Two Proportion Z-Test

For comparing two independent sample proportions [[Stattrek: Difference Between Proportions]][Stattrek, Hypothesis Test: Difference Between Proportions], [[PSU, 1]][Penn State, Applied Statistic: Comparing Two Population Proportions with Independent Samples], [[PSU, 2]][Penn State, Probability Theory and Mathematical Statistics: Comparing Two Proportions].
Assume two samples have 'success' proportions, 

$$\hat{p}_1 = \frac{x_1}{n_1} \ \ \textrm{and} \ \ \hat{p}_2 = \frac{x_2}{n_2}$$


the test statistic is

$$z = \frac{(\hat{p}_1-\hat{p}_2)-0}{\sqrt{\hat{p}_{pl}(1-\hat{p}_{pl})\Big(\frac{1}{n_1}+\frac{1}{n_2}\Big)}},$$

where 

$$\hat{p}_{pl} = \frac{x_1+x_2}{n_1+n_2}.$$

is the sample **pool proportion**.

For hypothesis testing, each sample needs to meet **success-failure** condition:

$$ n_1 \hat{p}_{pl} \ge 10, \ n_1(1 - \hat{p}_{pl}) ≥ 10,$$

and 

$$ n_2 \hat{p}_{pl} \ge 10, \ n_2(1 - \hat{p}_{pl}) ≥ 10.$$


## Reference

* [Stattrek, Hypothesis Test: Difference Between Proportions]: https://stattrek.com/hypothesis-test/difference-in-proportions.aspx
[[Stattrek: Difference Between Proportions] Stattrek, Hypothesis Test: Difference Between Proportions](https://stattrek.com/hypothesis-test/difference-in-proportions.aspx)

* [Penn State, Applied Statistic: Comparing Two Population Proportions with Independent Samples]: https://newonlinecourses.science.psu.edu/stat500/node/55/
[[PSU, 1] Penn State, Applied Statistic: Comparing Two Population Proportions with Independent Samples](https://newonlinecourses.science.psu.edu/stat500/node/55/)

* [Penn State, Probability Theory and Mathematical Statistics: Comparing Two Proportions]: https://newonlinecourses.science.psu.edu/stat414/node/268/
[[PSU, 2] Penn State, Probability Theory and Mathematical Statistics: Comparing Two Proportions](https://newonlinecourses.science.psu.edu/stat414/node/268/)

* [Stattrek, Hypothesis Test for a Proportion]: https://stattrek.com/hypothesis-test/proportion.aspx
[[Stattrek: Hypothesis Test for a Proportion] Stattrek, Hypothesis Test for a Proportion](https://stattrek.com/hypothesis-test/proportion.aspx)

* [Hypothesis Testing]: https://www.statisticshowto.com/probability-and-statistics/hypothesis-testing/
[[Statistics How to] Hypothesis Testing](https://www.statisticshowto.com/probability-and-statistics/hypothesis-testing/)