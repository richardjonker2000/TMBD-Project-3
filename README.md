# TMBD-Project-3

### **Members**
- Ana...
- Prosper Ablordeppey (106382)
- Richrd Jonker (109560)
- Tiago ...
## **Question 1:** <a id=Q1></a> 

For a metric in deciding whether a change in the algorithm provides an improvement or not, we decided that using an 
accuracy function would be most appropriate. After investigating, we found the purity accuracy function [[1]](#1):

$$Pur(Q,Y) = \frac{1}{N}\sum_{k=1}^K max_{j=1...c} |Q_k \cap Y_j |$$

Where $Q$ is the predicted class and $Y$ is the expected class. Essentially for each assigned cluster, we identify the most
common item in the cluster and assign the cluster the value of this item. We then get the accuracy by 
dividing the correctly clustered items with the total number of clusters. 

We chose an external measurement as they are easier to interpreter than the internal metrics mentioned in the work.
## **Question 2:**
After investigating various learning rate functions suggested by W. Natita et. al. [[2]](#2), we used the following learning 
rate functions in addition to the default function:

$$ \alpha(t,T) = \alpha(0) \times \exp\left(\frac{-t}{T}\right) \tag{default} $$

$$ \alpha(t) = \alpha(0) \times \frac{1}{t} \tag{linear}$$

$$ \alpha(t,T) = \alpha(0) \times \left(1 - \frac{t}{T}\right) \tag{inverse of time}$$

$$ \alpha(t,T) = \alpha(0) \times \exp{\left(\frac{t}{T}\right)} \tag{power series} $$

$T$ is the total number of iterations (epochs), and $t$ is the current iteration. These changes were integrated into the $decay\_learning\_rate$ functions specified above. We also investigated changing the initial learning rate by considering varying number of epochs in **[5, 10, 20, 30, 40, 50]** to better see the effect of the learning rate and the function changes. The learning rates we tested were **[1, 0.1, 0.001, 0.0001]**.

In summary,

<div align="center">

| Rank  |  Accuracy | Learning Function  | Learning Rate | Epochs
|---|---|---|---|---|
| 1st Highest  |  91\% | power  | 0.01  | 50
|---|---|---|---|---|
| 2nd Highest  | 90\%  | default & inverse |  1.0 \& 0.001 | 30 \& 10
|---|---|---|---|---|
| Lowest | 68\%  |  linear | 0.001  | 50  |

</div>
<br>

> Default

- With the default learning function, we observed a marginal increase in accuracy using smaller learning rates across all epochs.
- we observed a study increase in accuracy across epochs until 30 epochs where we recorded the highest.
- at learning rates 0.1 and 0.01, we do not observe any significant change in accuracy across the epochs considered
- with learning rate 0.001 we recorded second highest accuracy of 90\% for 30 iterations which later dropped to 78\% at 50 epochs.

> Linear Function

- The overall trend of accuracy tends to be decreasing with smaller learning rates, holding an epoch constant, 
- At learning rate 1, after 50 iteractions and more, the linear learning function does not produce any change in accuracy across all epochs.
- we realize a sudden increase in accuracy from 80\% to 86\% across the epochs at learning rate 0.1.
- we see an oscilating behaviour across epochs at learning rate 0.01 with 82\% minimum and 89\% maximum.
- Finally, at 0.001, the trend in accuracy seems to be decreasing across the epochs from 88\% at 20 epoch to 68\% at 50 epochs.

> Inverse Function

- keeping constant the epochs, we observe significant improvements in accuracy measure using smaller learning rates.
- At learning rate of 1.0, we see a steady decrease in the accuracy falling from 84\% at 30 epochs to 72\% at 500 epochs.
- With learning rate 0.1, there is a gradual decrease in accuracy from 86\% to 85\% across the epochs.
- Changing the number of epochs resulted in alot of improvements in the accuracy values recorded, although we had a drop at 100 epoch from 82\% to 76\%, the accuracy remained constant after hitting back at 86\%.
- After rising from 88\% to 90\% at 10 epochs using learning rate of 0.001, the accuracy remained constant steadily constant after dropping to 86\%.

> Power Function

- Using this function recorded the highest accuracy recorded. 
- using smaller learning rates for all epochs guarantees improvements.
- This function performed well on our training set compared to the others guranteeing a steady improvement.


From this we can see that improvements are obtainable by using different learning rate functions, however some of them perform better than others. 


http://www.cis.hut.fi/somtoolbox/documentation/somalg.shtml
## **Question 3**
We interpreted the curve of distribution as being the radius of the neighbourhood function. We investigated the use of 
removing the decreasing radius function, and also providing a fixed decreasing radius value of 0.5. We were not able to 
find substitutes to this function. All the tests were run with 40 epochs as this shows the best variation in the data. 
It was interesting to note the fixed decreasing radius had better results in 60 epochs than the dynamic decreasing radius.
Not changing the radius performed badly as expected.
## **Question 4**
We interpreted the normal distribution as the neighborhood function $h_{ck}(t)$. After investigating we found various possible neighborhood functions namely gaussian(the default one), bubble [[2]](#2), mexican-hat [[3]](#3) and triangular neighborhood.
We selected the mexican-hat neighborhood function as we were not able to understand how to implement the simple bubble 
function. The mexican-hat function is defined as [[3]](#3):

$$h_{ck}(t) =\left(1-2\frac{||r_k-r_c||^2}{\sigma(t)^2}\right) \times e^{-\frac{||r_k-r_c||^2}{\sigma(t)^2}}$$

With $x$ and $c$ representing an input pattern and centre, and width being $2w$. 

We also decided to investigate some changes to the gaussian neighbourhood function[4]:

$$h_{ck}(t) = e^{-q\frac{||r_k-r_c||^p}{\sigma(t)^2}}$$

Where the values of $p$ were  2 and 3, and the values of $q$ were 0.1,1 and 2.

When comparing the 2 base functions, we tested different ranges of epochs: 60,80,100,150 and 200.

We found that using a different neighbourhood function did not lead to a better result, however the graphs of the 
mexican-hat function looks slightly worse, due to the fact that there are more clusters with elements in. This is a 
weakness in the metric we picked in  [Question 1](#Q1). 

https://www.intechopen.com/chapters/69305
## **Question 5**
Let us consider $g(w) = w_k(t)+\alpha(t)h_{ck}(t)[x(t)-w_k(t)]$, where $\alpha \in ]0,1[$ represents the learning rate. So, $g'(w) = 1 - \alpha(t)h_{ck}(t)$. The fixed point method says that if $0 < max |g'(w)| < 1$, $g$ has an unique fixed point in $\mathbb{R}$ and $w_k = g(w_{k-1})$, with $k \in \mathbb{N}$ converges to that unique fixed point, $\forall w_0$ (initial approximation considered). So, the condition we have is: $0 < max |1 - \alpha(t)h_{ck}(t)| < 1$. Now, we will prove it. We know that $max |1 - \alpha(t)h_{ck}(t)|$ is achieved when $\alpha(t)h_{ck}(t)$ is minimum. We also know that $h_{ck}(t) = exp(-||r_k - r_c||^2 / \sigma(t) ^2) = \frac{1}{exp(||r_k - r_c||^2 / \sigma(t) ^2)}$. Obviously, $h_{ck}(t) > 0$. The maximum of $h_{ck}(t)$ is achieved when $r_k = r_c$, meaning that the maximum of the function is 1. For this reason, $h_{ck}(t) \in ]0,1]$. As we are interested to get $max |1 - \alpha(t)h_{ck}(t)|$, we need to give $\alpha(t)$ and $h_{ck}(t)$ very small values, very close to zero. Also, we can see that $\alpha(t)h_{ck}(t) \neq 1$, $\forall \alpha(t), h_{ck}(t)$. From here we get that $0 < max |1 - \alpha(t)h_{ck}(t)| < 1$, with $w \in \mathbb{R}$. In this way, the convergence is guaranteed.
## **Question 6**

The absolute errors for the approximations using the Euler approach are presented in brackets '()' beside the purity accuracy in the result output for the corresponding number of epochs. Accuracy as reported earlier increases across larger epochs aswell.
## **Question 7**

The adopted Runge-Kutta second order equation [[5]](#5) used is defined as 
\begin{align*}
w_{n+1} &= w_n + \alpha(t)h(t)[k_1+k_2] \\
k_2 &= \left(x+\frac{h(t)}{2} - w_n+\frac{k_1}{2}\right) \\ 
k_1 &= \frac{h(t)}{2}(x-w_n)
\end{align*}

We realized that, upon comparing the runge-kutta second order method with the implemented euler accuracies, the Runge-Kutta method generally performed better with higher accuracies. The estimated absolute error reduces significantly for smaller learning rates.

## **Question 8**

Just as the absolute errors for the euler method, the errors for the Runge-Kutta approximation is also presented in '()' beside the purity accuracy. These errors in general tend to be much smaller than those recorded from the Euler approach. In general, these errors decreased with smaller learning rates except very small learning rate with initial value 0.001.
## **Question 9**

- The Runge-Kutta approximation was very efficient in clustering the datapoints as compared to the Euler's approach as seen in the presented results. 
- Learning rates decay functions which seemed to be very effective were the power functions with learning rates 0.1 and 0.01 outperforming the others, with steady improvements for larger epochs. With very small initial learning rate of 0.001, we realized unexpected accuracies and corresponding higher error rate across all epochs considered. 
- For a more improved clustering, we infer that, keeping the neighbourhood radius below one (1) has significant impact on ensuring convergence of our optimization. Though at some specific parameters the Euler method performed well, we recommend the Runge-Kutta method for a more improved optimization.
## **References**
<a id=1>[1]</a> F. Forest, M. Lebbah, H. Azzag, and J. Lacaille, ‘A Survey and Implementation of Performance Metrics for Self-Organized Maps’, arXiv:2011.05847 [cs], Nov. 2020, Accessed: Jan. 30, 2022. [Online]. Available: http://arxiv.org/abs/2011.05847

<a id=2>[2]</a> Mathematics Department, King Mongkut’s University of Technology Thonburi, Bangkok, Thailand, W. Natita, W. Wiboonsak, and S. Dusadee, ‘Appropriate Learning Rate and Neighborhood Function of Self-organizing Map (SOM) for Specific Humidity Pattern Classification over Southern Thailand’, IJMO, vol. 6, no. 1, pp. 61–65, 2016, doi: 10.7763/IJMO.2016.V6.504.

<a id=3>[3]</a> https://coursepages2.tuni.fi/tiets07/wp-content/uploads/sites/110/2019/01/Neurocomputing3.pdf

<a id=4>[4]</a> L. A. Tu, Improving Feature Map Quality of SOM Based on Adjusting the Neighborhood Function. IntechOpen, 2019. doi: 10.5772/intechopen.89233.

<a id=5>[5]</a> https://mathforcollege.com/nm/mws/gen/08ode/mws_gen_ode_txt_runge2nd.pdf