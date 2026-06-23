# Canary Test

## The One-Sided KS Test

The KS test absolutely supports one-sided hypotheses. It does this by modifying the KS statistic, which is based on the Empirical Cumulative Distribution Function (eCDF). 

Let $F_{baseline}(x)$ be the eCDF of the current release, and $F_{canary}(x)$ be the eCDF of the new release. 

* Two-Sided (Did it change?): Uses the absolute maximum distance between the curves.

$$D = \sup_x |F_{baseline}(x) - F_{canary}(x)|$$

* One-Sided "Smaller is Better" (e.g., Latency, App Launch Time): If the canary is faster, its CDF will rise to $1.0$ quicker than the baseline. It will sit above and to the left of the baseline CDF. We test for this using the $D^-$ statistic:

$$D^- = \sup_x (F_{canary}(x) - F_{baseline}(x))$$

* One-Sided "Larger is Better" (e.g., Time to Battery Depletion): If the canary has better battery life, its values are larger, so its CDF is shifted to the right (sitting below the baseline). We test for this using the $D^+$ statistic:

$$D^+ = \sup_x (F_{baseline}(x) - F_{canary}(x))$$

If your one-sided test statistic is larger than the critical value, you reject the null hypothesis and conclude the canary is definitively better (or worse, depending on how you structure the test).


## The Mann-Whitney U Test

While the one-sided KS test is valid, it has a notable flaw in production environments: **it is highly sensitive to changes in the shape and variance of the distribution, not just the median. If the new software release has the exact same median latency, but a slightly longer tail (higher variance), the KS test might flag it as a massive difference.**

For canary testing where you want to know "Is the typical user experiencing a worse metric?", the industry standard non-parametric test is the **Mann-Whitney U Test** (also called the **Wilcoxon rank-sum test**).
How it works: 
* It combines all the data from both the baseline and canary, ranks them from smallest to largest, and then checks if the **ranks** from the canary tend to be significantly lower or higher than the baseline.
* One-Sided Application: You can easily run a one-sided Mann-Whitney test to ask: "What is the probability that a randomly selected iOS device on the Canary build has a higher battery drain than a randomly selected device on the Baseline build?" 


## In Production Reality

### 1. The "Big Data" Trap: Statistical vs. Practical Significance

At Apple’s scale (evaluating millions of devices), any standard statistical test—including the Mann-Whitney U test—will almost certainly return a highly significant p-value ($p < 0.001$) even if the CPU usage only increases by a microscopic $0.05\%$.

When your $N$ (sample size) is massive, **everything is statistically significant**.

Therefore, we cannot just rely on a p-value to make a launch decision. We must define Practical Significance.

### 2. Using "Non-Inferiority" Margins ($\delta$)

Since you know the new feature uses more memory, you don't test if memory usage is identical. You ask Engineering: "What is the memory budget for this feature?" Thus, you don't always need the canary to be better; you usually just need to prove it is not worse. This is called **Non-Inferiority Testing**.

Let's say Engineering expects a 15MB increase in RAM usage.
* The Wrong Test: Is Canary RAM = Baseline RAM? (Mann-Whitney will say no; rollout blocked).
* The Right Test becomes: Is Canary RAM $\le$ Baseline RAM + 15MB?

You shift the Baseline data by your tolerance margin ($\delta = 15\text{MB}$), and then run your one-sided Mann-Whitney U test. You are statistically proving that the memory increase is confined to the expected budget and isn't a runaway memory leak.

### 3. The Concept of "Guardrail Metrics"

CPU and Memory are usually Proxy Metrics. Users don't actually care about CPU utilization; they care about what high CPU utilization causes.

If you are defending a release that increases CPU usage, you must introduce Guardrail Metrics to the interviewer. You tell them: "We accept a 2% expected hit to CPU, provided our guardrail metrics remain flat."

* Examples of OS-level guardrails:Thermal Throttling Events: Did the device get so hot it had to slow itself down?
* OOM (Out of Memory) Crashes: Did our 15MB memory increase cause background apps (like Spotify or Maps) to be aggressively killed by the iOS Jetsam process?
* Battery Depletion Rate: Did the CPU increase actually translate to a noticeable drop in battery life for the average user?
* UI Frame Drops: Did the UI stutter while scrolling?


