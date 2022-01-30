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

$$1. \alpha(t) = \alpha(0) \times \frac{1}{t}$$

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





## References

[1]F. Forest, M. Lebbah, H. Azzag, and J. Lacaille, ‘A Survey and Implementation of Performance Metrics for Self-Organized Maps’, arXiv:2011.05847 [cs], Nov. 2020, Accessed: Jan. 30, 2022. [Online]. Available: http://arxiv.org/abs/2011.05847

[2]Mathematics Department, King Mongkut’s University of Technology Thonburi, Bangkok, Thailand, W. Natita, W. Wiboonsak, and S. Dusadee, ‘Appropriate Learning Rate and Neighborhood Function of Self-organizing Map (SOM) for Specific Humidity Pattern Classification over Southern Thailand’, IJMO, vol. 6, no. 1, pp. 61–65, 2016, doi: 10.7763/IJMO.2016.V6.504.


