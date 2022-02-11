from matplotlib import pyplot as plt
from matplotlib import patches as patches

__author__ = 'Shishir Adhikari'
import numpy as np
import pandas as pd

# np.set_printoptions(precision=16)


class SOM:
    """
    Python implementation of online SOM using numpy
    Training Algorithm:
    ------------------------------------------------------------
    initialize weight vectors
    for (epoch = 1,..., Nepochs)
        t = 0
        interpolate new values for α(t) and σ (t)
            for (record = 1,..., Nrecords)
                t = t + 1
                for (k = 1,..., K)
                    compute distances dk using Eq. (1)
                end for
                compute winning node c using Eq. (2)
                for (k = 1,..., K)
                    update weight vectors wk using Eq. (3)
                end for
            end for
    end for

    Equation 1) dk (t) = [x(t) − wk(t)]^2
    Equation 2) dc(t) ≡ min dk(t)
    Equation 3) wk (t + 1) = wk (t) + α(t)hck(t)[x(t) − wk (t)]
    where, hck(t) = exp(−[rk − rc]^2 / σ (t)^2)

    ----------------------------------------------------------------
    """

    def __init__(self, net_x_dim, net_y_dim, num_features):
        """

        :param net_x_dim: size of net (x)
        :param net_y_dim:  size of net (y)
        :param num_features: number of features in input data
        :return:
        """
        self.network_dimensions = np.array([net_x_dim, net_y_dim])
        self.init_radius = min(self.network_dimensions[0], self.network_dimensions[1])
        # initialize weight vectors
        self.num_features = num_features
        self.initialize()

    def initialize(self):
        self.net = np.random.random((self.network_dimensions[0], self.network_dimensions[1], self.num_features))
    
    def set_weights(self, weights):
        """sets the weight for a given experiment

        Args:
            weights (np.ndarray): network weights to be used for computation
        """
        self.net = weights

    def train(self, data, pq=(2,1), num_epochs=1, init_learning_rate=0.01, lr_decay_function='default', radius_decay_function='fixed', influence_function='gaussian', resetWeights=False, show_plot=False):
        """
        :param data: the data to be trained
        :param num_epochs: number of epochs (default: 100)
        :param init_learning_rate: initial learning rate (default: 0.01)
        :return:
        """
        if resetWeights:
            self.initialize()
        num_rows = data.shape[0]
        indices = np.arange(num_rows)
        self.time_constant = num_epochs / np.log(self.init_radius)

        # visualization
        if show_plot:
            fig = plt.figure()
        else:
            fig = None
        
        network_grid = self.net.copy()

        for i in range(1, num_epochs + 1):
            
            # interpolate new values for α(t) and σ (t)
            radius = self.decay_radius(i, deviation_curve=radius_decay_function)
            learning_rate = self.decay_learning_rate(initial_learning_rate=init_learning_rate, iteration=i, num_iterations=num_epochs, lr_decay_function=lr_decay_function)
            
            # visualization
            # vis_interval = int(num_epochs/10)
            # if i % vis_interval == 0:
            #     if fig is not None:
            #         self.show_plot(fig, i/vis_interval, i)
            #     print("SOM training epoches %d" % i)
            #     print("neighborhood radius ", radius)
            #     print("learning rate ", learning_rate)
            #     print("-------------------------------------")

            # shuffling data
            np.random.shuffle(indices)

            # for (record = 1,..., Nrecords)
            for record in indices:
                row_t = data[record, :]

                # find its Best Matching Unit
                bmu, bmu_idx = self.find_bmu(row_t)
                # for (k = 1,..., K)
                for x in range(self.network_dimensions[0]):
                    for y in range(self.network_dimensions[1]):
                        weight = network_grid[x, y, :].reshape(1, self.num_features)
                        w_dist = np.sum((np.array([x, y]) - bmu_idx) ** 2)

                        # if the distance is within the current neighbourhood radius
                        if w_dist <= radius ** 2:
                            # update weight vectors wk using Eq. (3m )
                            influence = SOM.calculate_influence(w_dist, radius, function=influence_function, pq=pq)
                            
                            new_w = weight + (learning_rate * influence * (row_t - weight))
                            network_grid[x, y, :] = new_w.reshape(1, self.num_features)

        if fig is not None:
            plt.show()
        
        self.net = network_grid


    def train_p(self, data, pq=(2,1), num_epochs=1, init_learning_rate=0.01, lr_decay_function='default', radius_decay_function='fixed', influence_function='gaussian', resetWeights=False, show_plot=False):
        """A vectorized computation. Faster compared to the use of the for loops

        Args:
            data (np.ndarray): data matrix of samples
            pq (tuple, optional): parameters to compute influence. Defaults to (2,1).
            num_epochs (int, optional): number of iterations over the dataset. Defaults to 1.
            init_learning_rate (float, optional): rate of convergence. Defaults to 0.01.
            lr_decay_function (str, optional): function to use to decay learning rate. Defaults to 'default'.
            radius_decay_function (str, optional): function used to decay radius. Default to fixed.
            influence_function (str, optional): function to compute influence with. Defaults to 'gaussian'.
            resetWeights (bool, optional): whether to fetch new weights or not. Defaults to False.
            show_plot (bool, optional): whether to display plot or not. Defaults to True.

        Returns:
            np.ndarray: weights of neural network.
        """
        # reset weights to new random weights
        if resetWeights:
            self.initialize()
        
        # get length og dataset
        num_rows = data.shape[0]
        indices = np.arange(num_rows)

        self.time_constant = num_epochs / np.log(self.init_radius)

        # visualization
        if show_plot:
            fig = plt.figure()
        else:
            fig = None

        for i in range(1, num_epochs + 1):
            
            # interpolate new values for α(t) and σ (t)
            radius = self.decay_radius(iteration=i, deviation_curve='exponential')
            learning_rate = self.decay_learning_rate(initial_learning_rate=init_learning_rate, iteration=i, num_iterations=num_epochs, lr_decay_function=lr_decay_function)
            
            #  visualization
            # vis_interval = int(num_epochs/10)
            # if i % vis_interval == 0:
            #     if fig is not None:
            #         self.show_plot(fig, i/vis_interval, i)
            #     print("SOM training epoches %d" % i)
            #     print("neighborhood radius ", radius)
            #     print("learning rate ", learning_rate)
            #     print("-------------------------------------")

            np.random.shuffle(indices)  # shuffling data

            for row_t in data:

                # get bmu for current sample (competition)
                bmu, bmu_idx = self.find_bmu(row_t)

                # compute the distances between neurons. Get the collaborating neurons (those in defined radius)
                w_dist = np.array([np.sum((np.array(neuron_index) - bmu_idx)**2) for neuron_index in np.ndindex(self.network_dimensions[0],self.network_dimensions[1])]).reshape(self.network_dimensions[0],self.network_dimensions[1]) 
                w_dist = pd.DataFrame(w_dist).apply(lambda k: pd.Series([_ if (_ <= radius ** 2) else 0 for _ in k]))
                w_dist = w_dist.fillna(0).values
                
                # compute influence between collaborating neurons and BMU neuron
                influence = SOM.calculate_influence(distance=w_dist, radius=radius, function=influence_function, pq=pq)
                
                # update weights of neurons; affecting collaborating neurons and BMU (Updates)
                self.net = self.net + (learning_rate*influence*(row_t - self.net))

        if fig is not None:
            plt.show()


    @staticmethod
    def calculate_influence(distance, radius, function, pq=(2,1)):
        """
        radius of influence for collaboration.

        Args:
            distance (float): distance between input vector and neuron. ['mexican_hat', 'gaussian']
            radius (float): radius of neighbour inclusion
            function (str): ['mexican_hat', gaussian]
            q ([type]): [description]
            p ([type]): [description]

        Returns:
            float: 
        """
        if function == "mexican_hat":
            return  (1 - distance / (radius ** 2)) * np.exp(- distance / (2 * (radius ** 2)))
        
        return np.exp(-pq[1] * distance / (2 * (radius ** pq[0])))


    def find_bmu(self, row_t):
        """
        Competition Stage
        Find the Best Matching Unit (BMU) for a given vector, row_t, in the SOM
        
        Returns:
            bmu, bmu_idx (tuple):
                bmu     - the high-dimensional Best Matching Unit
                bmu_idx - is the index of this vector in the SOM
        """

        distances = np.sum((self.net - row_t)**2, axis=2)  # compute the euclidean distance of sample from each neuron
        bmu_idx = np.unravel_index(np.argmin(distances, axis=None), distances.shape)  # fetch minimum distanced neuron
        bmu_weight = self.net[bmu_idx]  # get BMU neuron's corresponding weights
        return bmu_weight, np.array(bmu_idx)


    def predict(self, data):
        """
        finds the best matching using of the given data matrix.

        Args:
            data (nd.array): matrix of samples to cluster.

        Returns:
            bmu, bmu_ids:
                bmu     - weights associated with the best matching neuron/cluster for each sample.
                bmu_idx - positions of best matching neuron in the grid.
        """
        bmu, bmu_idx = self.find_bmu(data)
        return bmu, bmu_idx
    
    def decay_radius(self, iteration, deviation_curve):
        """
        reduce neighbourhood radius for each iteration via given decay_curve

        Args:
            iteration (int): current iteration
            decay_curve (str): decay curve to use for the reduction. could be ['eponential', 'linear', 'fixed']

        Returns:
            float: the computed radius for current iteration
        """
        if deviation_curve == "exponential":  # exponential decay
            return self.init_radius * np.exp(-iteration / self.time_constant)
        elif deviation_curve == "linear":  # linear decay
            return self.init_radius * 0.5
        return self.init_radius # fixed radius


    def decay_learning_rate(self, initial_learning_rate, iteration, num_iterations, lr_decay_function='default'):
        """
        reduce learning rate wrt the decay function

        Args:
            initial_learning_rate (float): base learning rate
            iteration (int): current iteration
            num_iterations (int): total number of epochs
            lr_decay_function (str): the function used to deacay lr

        Returns:
            float: computed learning rate
        """
        if lr_decay_function == "linear":
            return initial_learning_rate*(1/iteration)
        elif lr_decay_function == "inverse":
            return initial_learning_rate*(1-iteration/num_iterations)
        elif lr_decay_function == "power":
            return initial_learning_rate * np.exp(-iteration / num_iterations)
        # for default or undefined
        return initial_learning_rate * np.exp(iteration / num_iterations)


    def show_plot(self, fig, position, epoch):
        # setup axes
        ax = fig.add_subplot(2, 5, position, aspect="equal")
        ax.set_xlim((0, self.net.shape[0] + 1))
        ax.set_ylim((0, self.net.shape[1] + 1))
        ax.set_title('Ep: %d' % epoch)

        # plot the rectangles
        for x in range(1, self.net.shape[0] + 1):
            for y in range(1, self.net.shape[1] + 1):
                ax.add_patch(patches.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                               facecolor=self.net[x - 1, y - 1, :],
                                               edgecolor='none'))