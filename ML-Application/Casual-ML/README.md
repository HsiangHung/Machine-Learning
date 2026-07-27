# Casual Inference

 Causal inference focuses on determining whether there is a cause-and-effect relationship between a specific action (the “treatment”) and its result (the “outcome”).


## Decision Workflow 

If we decide to make treatment or not, using

$$\hat{\tau} \times \textrm{LTV} - c > 0,$$

where 

$$ \hat{\tau} = P(\textrm{renew} | \textrm{treatment}) − P(\textrm{renew} | \textrm{no treatment}). $$

<img src="https://github.com/HsiangHung/Machine-Learning/blob/master/ML-Application/Casual-ML/images/CLV_workflow.png" width="900">