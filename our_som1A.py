from matplotlib import pyplot as plt
from matplotlib import patches as patches

__author__ = 'Shishir Adhikari'
import numpy as np


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
        """sets up new grid of network grid weights for optimization.
        """
        self.net = np.random.random((self.network_dimensions[0], self.network_dimensions[1], self.num_features))


    def set_weights(self, weights):
        """sets weight to use for SOM

        Args:
            weights (ndarray): network grid of weights for neurons.
        """
        self.net = weights


    def train(self, data, lr_decay_function, num_epochs=1, init_learning_rate=0.01, resetWeights=False, show_plot=False, method='euler'):
        """wrapper for training SOM using euler or runge-kutta method

        Args:
            data (ndarray): matrix of samples as rows
            lr_decay_function (str): learning rate decay function
            num_epochs (int, optional): total number of epochs. Defaults to 1.
            init_learning_rate (float, optional): initial learning rate. Defaults to 0.01.
            resetWeights (bool, optional): whether to generate new weights or use existing. Defaults to False.
            show_plot (bool, optional): whether to show plots or not. Defaults to False.
            method (str, optional): optimization method. euler or runge-kutta. Defaults to 'euler'.
        """
        self.lr_decay_function = lr_decay_function
        self.num_epochs = num_epochs
        self.init_learning_rate = init_learning_rate
        self.resetWeights = resetWeights
        self.show_plot = show_plot
        self.data = data

        if method == 'runge-kutta':
           self.runge_kutta()
        else:  # euler method
            self.euler()


    def euler(self):
        """
        weight updates in self.net using the euler equation
            - w_n+1 = wn + a*h(x - wn)

        weights update in self.approx_net using the approximation formular (solution to the ode)
            - wn+1 = (wn + wn*x)/(1 + x), 
        """
        # reset weight if the need be
        if self.resetWeights:
            self.initialize()
        num_rows = self.data.shape[0]
        indices = np.arange(num_rows)
        self.time_constant = self.num_epochs/np.log(self.init_radius)
        self.approx_net = self.net.copy()  # w0 for approximation

        # visualization
        if self.show_plot:
            fig = plt.figure()
        else:
            fig = None

        # for (epoch = 1,..., Nepochs)
        for i in range(1, self.num_epochs + 1):
            # interpolate new values for α(t) and σ (t)
            radius = self.decay_radius(i)
            learning_rate = self.decay_learning_rate(iteration=i)
            # visualization
            #vis_interval = int(self.num_epochs/10)
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
                row_t = self.data[record, :]

                # find its Best Matching Unit
                bmu, bmu_idx = self.find_bmu(row_t)
                bmu_ea, bmu_idx_ea = self.find_bmu(row_t, self.approx_net)  # for euler approximation

                # for (k = 1,..., K)
                for x in range(self.network_dimensions[0]):
                    for y in range(self.network_dimensions[1]):

                        # weight update for euler equation
                        weight = self.net[x, y, :].reshape(1, self.num_features)
                        w_dist = np.sum((np.array([x, y]) - bmu_idx) ** 2)
                        # if the distance is within the current neighbourhood radius
                        if w_dist <= radius ** 2:
                            # update weight vectors wk using Eq. (3)
                            influence = SOM.calculate_influence(w_dist, radius)
                            new_w = weight + (learning_rate * influence * (row_t - weight))
                            self.net[x, y, :] = new_w.reshape(1, self.num_features)

                        # weight-update using euler approximation function 
                        weight_approx = self.approx_net[x, y, :].reshape(1, self.num_features)
                        w_dist_approx = np.sum((np.array([x, y]) - bmu_idx_ea) ** 2)
                        if w_dist_approx <= radius ** 2:
                            w0 = self.approx_net[x, y, :].reshape(1, self.num_features)
                            dw = (w0 + w0*x)/(1+x)
                            self.approx_net[x, y, :] = dw

        if fig is not None:
            plt.show()

    def runge_kutta(self):
        """
        weight updates in self.net using the runge-kutta second order equation
            - w_n+1 = wn + a*h*k2
            - k2 = h*(x+h/2 - wn+k1/2)
            - k1 = x - wn

        weight updates in self.approx_net using the runge-kutta second order approximation formular (solution to the ode)
            - wn+1 = (wn + wn*x)/(1 + x), 
        """
        # reset weight if the need be
        if self.resetWeights:
            self.initialize()
        num_rows = self.data.shape[0]
        indices = np.arange(num_rows)
        self.time_constant = self.num_epochs / np.log(self.init_radius)
        self.approx_net =self.net.copy()  # w0 for approximation

        # visualization
        if self.show_plot:
            fig = plt.figure()
        else:
            fig = None

        # for (epoch = 1,..., Nepochs)
        for i in range(1, self.num_epochs + 1):
            # interpolate new values for α(t) and σ (t)
            radius = self.decay_radius(i)
            learning_rate = self.decay_learning_rate(iteration=i)
            # visualization
            #vis_interval = int(num_epochs/10)
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
                row_t = self.data[record, :]

                # find its Best Matching Unit
                bmu, bmu_idx = self.find_bmu(row_t)
                bmu_ra, bmu_idx_ra = self.find_bmu(row_t, weights=self.approx_net)  # runge-kutta approximation

                # for (k = 1,..., K)
                for x in range(self.network_dimensions[0]):
                    for y in range(self.network_dimensions[1]):

                        # weight update for runge-kutta equation
                        weight = self.net[x, y, :].reshape(1, self.num_features)
                        w_dist = np.sum((np.array([x, y]) - bmu_idx) ** 2)
                        if w_dist <= radius ** 2: # if the distance is within the current neighbourhood radius
                            influence = SOM.calculate_influence(w_dist, radius)
                            # update weight vectors wk using runge-kutta equation of order 4
                            k1 = influence * (row_t - weight)  # f(x0,y0) = hf(x0,y0)
                            k2 = (row_t+influence/2 - weight+k1/2)  # h f [ xo + h/2 , yn + k1 / 2] eqn (2)
                            new_w = weight + learning_rate * influence * k2  # y_n+1 = yn + h*k2
                            self.net[x, y, :] = new_w.reshape(1, self.num_features)

                        # weight-update using rk approximation function 
                        weight_approx = self.approx_net[x, y, :].reshape(1, self.num_features)
                        w_dist_approx = np.sum((np.array([x, y]) - bmu_idx_ra) ** 2)
                        if w_dist_approx <= radius ** 2:
                            w0 = self.approx_net[x, y, :].reshape(1, self.num_features)
                            dw = (w0 + w0*x)/(1+x)
                            self.approx_net[x, y, :] = dw.reshape(1, self.num_features)

        if fig is not None:
            plt.show()    


    def decay_radius(self, iteration):
        """radius within which to consider neigboiuring neurons

        Args:
            iteration (int): current iteration

        Returns:
            float: computed radius. decays with every iteration.
        """
        return self.init_radius * np.exp(-iteration/self.time_constant)#rk-rx^2


    def decay_learning_rate(self, iteration):
        """decay learning rate wrt the decay function

        Args:
            iteration (int): current iteration

        Returns:
            float: computed learning rate
        """
        if self.lr_decay_function == "linear":
            return self.init_learning_rate*(1/iteration)
        elif self.lr_decay_function == "inverse":
            return self.init_learning_rate*(1-iteration/self.num_epochs)
        elif self.lr_decay_function == "power":
            return self.init_learning_rate * np.exp(-iteration/self.num_epochs)
        return self.init_learning_rate * np.exp(iteration/self.num_epochs)  # for default or undefined


    @staticmethod
    def calculate_influence(distance, radius): #distribution
        return np.exp(-distance / (2 * (radius ** 2)))#SD - 2 * r**2

    def find_bmu(self, row_t, weights=None):
        """
        Competition Stage
        Find the best matching unit for a given vector, row_t, in the SOM
        
        Returns:
            bmu, bmu_idx (tuple):
                bmu     - the high-dimensional Best Matching Unit
                bmu_idx - is the index of this vector in the SOM
        """
        if weights is None:
            weights = self.net
        distances = np.sum((weights - row_t)**2, axis=2)  # compute the euclidean distance of sample from each neuron
        bmu_idx = np.unravel_index(np.argmin(distances, axis=None), distances.shape)  # fetch minimum distanced neuron
        bmu_weight = weights[bmu_idx]  # get BMU neuron's corresponding weights
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