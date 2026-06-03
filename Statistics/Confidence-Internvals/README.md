
# Confidence Intervals (CI)

## CI for One Sample

$$\textrm{CI} = \bar{x} \pm z^* \frac{s}{\sqrt{n}},$$

where $\bar{x}$ is the mean and $s$ is the standard deviation. 

For 95% CI, $z^* = 1.96$. Higher confidence, we have a wider interval.

## CI for Two Independent Samples, Continuous Outcome

[Confidence Interval for Two Independent Samples, Continuous Outcome](https://sphweb.bumc.bu.edu/otlt/mph-modules/bs/bs704_confidence_intervals/bs704_confidence_intervals5.html), 
[[Kent State University]][SPSS TUTORIALS: INDEPENDENT SAMPLES T TEST], [[JMP]][The Two-Sample t-Test], [[UF Biostatistics]][Two Independent Samples]

### Equal variance is assumed

When the two **independent** samples are assumed to be drawn from populations with identical population variances:

If $n_1 > 30$ or $n_2 > 30$, use the Z-table, the CI is

$$\big(\bar{x}_1 -\bar{x}_2\big) \pm z S_p \sqrt{\frac{1}{n_1}+\frac{1}{n_2}}.$$


If $n_1 < 30$ or $n_2 < 30$, use the t-table, the CI is

$$\big(\bar{x}_1 -\bar{x}_2\big) \pm t_{n_1+n_2-2} S_p \sqrt{\frac{1}{n_1}+\frac{1}{n_2}},$$

where the pooled standard deviation is

$$S_p =\sqrt{\frac{(n_1-1)s_1^2+(n_2-1)s_2^2}{n_1+n_2-2}}.$$


### Equal variances are not assume

When the two independent samples are assumed to be drawn from populations with unequal variances, the CI is  

$$\big( \bar{x}_1 - \bar{x}_2 \big) \pm t \sqrt{\frac{s_1^2}{n_1}+\frac{s_2^2}{n_2}}.$$


### Example 1

[[Terence Shin]][40 Statistics Interview Problems and Answers for Data Scientists] In a study of emergency room waiting times, investigators consider a new and the standard triage systems. To test the systems, administrators selected 20 nights and randomly assigned the new triage system to be used on 10 nights and the standard system on the remaining 10 nights. They calculated the nightly median waiting time (MWT) to see a physician. The average MWT for the new system was 3 hours with a variance of 0.60 while the average MWT for the old system was 5 hours with a variance of 0.68. Consider the 95% confidence interval estimate for the differences of the mean MWT associated with the new system. 

**Assume a constant variance.** What is the interval? Subtract in this order (New System — Old System).

$$\textrm{CI} = \big( 3 - 5 \big) \pm t_{18} \sqrt{\frac{(9*0.6^2+9*0.68^2)}{18}} \sqrt{\frac{2}{10}} = -2\pm 2.101*0.352.$$



### Example 2

[[Terence Shin]][40 Statistics Interview Problems and Answers for Data Scientists] To further test the hospital triage system, administrators selected 200 nights and randomly assigned a new triage system to be used on 100 nights and a standard system on the remaining 100 nights. They calculated the nightly median waiting time (MWT) to see a physician. The average MWT for the new system was 4 hours with a standard deviation of 0.5 hours while the average MWT for the old system was 6 hours with a standard deviation of 2 hours. 

Consider the hypothesis of a decrease in the mean MWT associated with the new treatment. 
What does the 95% independent group confidence interval with unequal variances suggest vis a vis this hypothesis? 

Because there’s so many observations per group, just use the Z quantile instead of the T.

$$\textrm{CI} = \big( 4 - 6 \big) \pm z^* \sqrt{\frac{(99*0.5^2+99*2^2)}{200-2}} \sqrt{\frac{2}{100}} = -2\pm 1.96*0.20506$$




## CI for Two Independent Samples, Dichotomous Outcome

The formula for the [confidence interval for the difference in proportions](https://sphweb.bumc.bu.edu/otlt/mph-modules/bs/bs704_confidence_intervals/bs704_confidence_intervals7.html) is:

$$\big( \hat{p}_1 - \hat{p}_2) \pm z \sqrt{\frac{\hat{p}_1(1-\hat{p}_1)}{n_1} + \frac{\hat{p}_2(1-\hat{p}_2)}{n_2}}.$$


## CI for t-distribution

To estimate the mean [[Courera - Inferential Statistics: Inference for a mean]][Inference for a mean]

$$ \textrm{point estimate} \pm \textrm{margin of error} = \bar{x} \pm t^*_{n-1} \frac{s}{\sqrt{n}}.$$

Use t-table to look up critical t score.

#### Reference

* [The Two-Sample t-Test]: https://www.jmp.com/en_us/statistics-knowledge-portal/t-test/two-sample-t-test.html
[[JMP] The Two-Sample t-Test](https://www.jmp.com/en_us/statistics-knowledge-portal/t-test/two-sample-t-test.html)

* [SPSS TUTORIALS: INDEPENDENT SAMPLES T TEST]: https://libguides.library.kent.edu/spss/independentttest
[[Kent State University] SPSS TUTORIALS: INDEPENDENT SAMPLES T TEST](https://libguides.library.kent.edu/spss/independentttest)

* [40 Statistics Interview Problems and Answers for Data Scientists]:https://towardsdatascience.com/40-statistics-interview-problems-and-answers-for-data-scientists-6971a02b7eee
[[Terence Shin] 40 Statistics Interview Problems and Answers for Data Scientists](https://towardsdatascience.com/40-statistics-interview-problems-and-answers-for-data-scientists-6971a02b7eee)

* [Two Independent Samples]:https://bolt.mph.ufl.edu/6050-6052/unit-4b/module-13/two-independent-samples/
[[UF Biostatistics] Two Independent Samples](https://bolt.mph.ufl.edu/6050-6052/unit-4b/module-13/two-independent-samples/)

* [Inference for a mean]:https://www.coursera.org/learn/inferential-statistics-intro/lecture/qs7Ml/inference-for-a-mean
[[Courera - Inferential Statistics: Inference for a mean] Inference for a mean](https://www.coursera.org/learn/inferential-statistics-intro/lecture/qs7Ml/inference-for-a-mean)


