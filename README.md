# TMBD-Project-3

## Question 1:

For a metric in deciding whether a change in the algorithm provides an improvement or not, we decided that using an 
accuracy function would be most appropriate. After investigating, we found the purity accuracy function[1]:

$$Pur(Q,Y) = \frac{1}{N}\sum_{k=1}^K max_{j=1...c} |Q_k \cap Y_j |$$

Where $Q$ is the predicted class and $Y$ is the expected class. Essentially for each assigned cluster, we identify the most
common item in the cluster and assign the cluster the value of this item. We then get the accuracy by 
dividing the correctly clustered items with the total number of clusters. 

We chose an external measurement as they are easier to interpreter than the internal metrics mentioned in the work.

## Question 2:

After investigating various learning rate functions suggested by W. Natita et. al.[2], we used the following learning 
rate functions in addition to the default function:

$$ 1. \alpha(t) = \alpha(0) \times \frac{1}{t} $$

$$2. \alpha(t,T) = \alpha(0) \times (1 - \frac{t}{T})$$

$$3. \alpha(t,T) = \alpha(0) \times e^{\frac{t}{T}}$$

The learning rate functions are Linear(1), inverse of time(2), and power series (3). $T$ is the number of iterations, 
and $t$ is the current iteration. These changes were integrated into the $decay_learning_rate$ function. We also 
investigated changing the initial learning rate and reduced the number of epochs to 100 to better see the effect of the 
learning rate and the function changes. The learning rates we tested were 1, 0.1, 0.001, and 0.0001.

In summary, the highest accuracy we obtained was 80.9\%, achieved by:
- default, 0.01
- power, 0.1

It is interesting to note that by using lower learning rates, the power learning rate function performed substantially
worse only achieving accuracies of around 40\%-50\%. This is probably due to the low number of epochs. The inverse 
learning rate function achieved 0.8 for all values of learning rates. All other accuracies in the default learning rate 
were 80\% and linear also performed well with accuracies ranging from 70\% to 80\%.

From this we can see that improvements are not really obtainable by using different learning rate functions, however 
some of them perform worse. 


http://www.cis.hut.fi/somtoolbox/documentation/somalg.shtml

## Question 3

We interpreted the curve of distribution as being the radius of the neighbourhood function. We investigated the use of 
removing the decreasing radius function, and also providing a fixed decreasing radius value of 0.5. We were not able to 
find substitutes to this function. all the tests were run with 40 epochs as this shows the best variation in the data. 
It was interesting to note the fixed decreasing radius had better results in 60 epochs than the dynamic decreasing radius.
Not changing the radius performed badly as expected.

## Question 4
We interpreted the normal distribution as the neighborhood function $h_{ck}(t)$. After investigating we found various 
possible neighborhood functions namely gaussian(the default one), bubble[2], mexican-hat[3] and triangular neighborhood.
We selected the mexican-hat neighborhood function as we were not able to understand how to implement the simple bubble 
function. The mexican-hat function is defined as [3]:

$$h_{ck}(t) =(1-2\frac{||r_k-r_c||^2}{\sigma(t)^2}) \times e^{-\frac{||r_k-r_c||^2}{\sigma(t)^2}}$$

With $x$ and $c$ representing an input pattern and centre, and width being $2w$. 

We also decided to investigate some changes to the gaussian neighbourhood function[4]:

$$h_{ck}(t) = e^{-q\frac{||r_k-r_c||^p}{\sigma(t)^2}}$$

Where the values of $p$ were  2 and 3, and the values of $q$ were 0.1,1 and 2.

When comparing the 2 base functions, we tested different ranges of epochs: 60,80,100,150 and 200.

We found that using a different neighbourhood function did not lead to a better result, however the graphs of the 
mexican-hat function looks slightly worse, due to the fact that there are more clusters with elements in. This is a 
weakness in the metric we picked in Question 1. 

https://www.intechopen.com/chapters/69305







## References

[1]F. Forest, M. Lebbah, H. Azzag, and J. Lacaille, ‘A Survey and Implementation of Performance Metrics for Self-Organized Maps’, arXiv:2011.05847 [cs], Nov. 2020, Accessed: Jan. 30, 2022. [Online]. Available: http://arxiv.org/abs/2011.05847

[2]Mathematics Department, King Mongkut’s University of Technology Thonburi, Bangkok, Thailand, W. Natita, W. Wiboonsak, and S. Dusadee, ‘Appropriate Learning Rate and Neighborhood Function of Self-organizing Map (SOM) for Specific Humidity Pattern Classification over Southern Thailand’, IJMO, vol. 6, no. 1, pp. 61–65, 2016, doi: 10.7763/IJMO.2016.V6.504.

[3] https://coursepages2.tuni.fi/tiets07/wp-content/uploads/sites/110/2019/01/Neurocomputing3.pdf

[4] L. A. Tu, Improving Feature Map Quality of SOM Based on Adjusting the Neighborhood Function. IntechOpen, 2019. doi: 10.5772/intechopen.89233.
