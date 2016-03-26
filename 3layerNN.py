# Package imports
import sys
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib
class Network:
    np.random.seed(0)
    l1_dimension = 2
    l3_dimension = 2
    eps = 0.01
    lamb = 0.01
    def __init__(self,tset_length, active_choice,hidden_choice):
      self.syn1 = np.random.randn(2, hidden_choice) / np.sqrt(2)
      self.syn2 = np.random.randn(hidden_choice, 2) / np.sqrt(2)
      self.back_prop_1 = np.zeros((1, hidden_choice))
      self.back_prop_2 = np.zeros((1, 2))
      self.activation_choice = active_choice
      self.tset_length = tset_length
      self.oL = None
      self.hidden_d = hidden_choice
    def forward_propagation(self,scatter_plot_values):
        iL = scatter_plot_values.dot(self.syn1) + self.back_prop_1  # Map input layer 2-nodes to hidden layer 3-nodes
        if self.activation_choice == 0:
            self.oL = np.tanh(iL)
            syn2_shift = self.oL.dot(self.syn2) + self.back_prop_2
            return np.exp(syn2_shift)
        elif activation_choice == 1:
            return np.sigmoid(iL)
        elif activation_choice == 2:
            return np.sin(iL)
    def back_propagation(self,x,class_membership,scatter_plot_values):
        syn2_mapping = x
        syn2_mapping[range(self.tset_length), class_membership] -= 1
        self.syn2_mapping = (self.oL.T).dot(syn2_mapping)
        self.bp2_mapping = np.sum(syn2_mapping, axis=0, keepdims=True)
        syn1_mapping = syn2_mapping.dot(self.syn2.T) * (1 - np.power(self.oL, 2))
        self.syn1_change = np.dot(scatter_plot_values.T, syn1_mapping)
        self.bp1_change = np.sum(syn1_mapping, axis=0)
        return
    def train(self,scatter_plot_values,epochs,class_membership):
        print_loss = True
        for i in xrange(0, epochs):
            predicted_scores = self.forward_propagation(scatter_plot_values)
            softM_input = predicted_scores / np.sum(predicted_scores, axis=1, keepdims=True)
            self.back_propagation(softM_input,class_membership,scatter_plot_values)
            self.syn2_mapping += self.lamb * self.syn2
            self.syn1_change += self.lamb * self.syn1
            self.syn1 += -self.eps * self.syn1_change
            self.back_prop_1 += -self.eps * self.bp1_change
            self.syn2 += -self.eps * self.syn2_mapping
            self.back_prop_2 += -self.eps * self.bp2_mapping
            if print_loss and i % 1000 == 0:
              print "Loss after iteration %i: %f" %(i, self.display_loss(scatter_plot_values,class_membership))
        return self.syn1, self.back_prop_1, self.syn2, self.back_prop_2
    def display_loss(self,scatter_plot_values,class_membership):
        syn2_shift = scatter_plot_values.dot(self.syn1) + self.back_prop_1
        syn1_map = np.tanh(syn2_shift)
        syn2_shift = syn1_map.dot(self.syn2) + self.back_prop_2
        network_prediction = np.exp(syn2_shift)
        softM_input = network_prediction / np.sum(network_prediction, axis=1, keepdims=True)
        softM_output = -np.log(softM_input[range(self.tset_length), class_membership])
        amount_of_loss = np.sum(softM_output)
        amount_of_loss += self.lamb/2 * (np.sum(np.square(self.syn1)) + np.sum(np.square(self.syn2)))
        return 1./self.tset_length * amount_of_loss
def prediction_function(quad, x):
    syn1, back_prop_1, syn2, back_prop_2 = quad
    syn2_shift = x.dot(syn1) + back_prop_1
    syn1_map = np.tanh(syn2_shift)
    syn2_shift = syn1_map.dot(syn2) + back_prop_2
    network_prediction = np.exp(syn2_shift)
    softM_input = network_prediction / np.sum(network_prediction, axis=1, keepdims=True)
    return np.argmax(softM_input, axis=1)
# ,tableau20
def draw_boundary(pred_func,scatter_plot_values,class_membership,m):

    # Ensure that the axis ticks only show up on the bottom and left of the plot.
    # Ticks on the right and top of the plot are generally unnecessary chartjunk.
    # ax.get_xaxis().tick_bottom()
    # ax.get_yaxis().tick_left()

    domain_min, domain_max = scatter_plot_values[:, 0].min() - .5, scatter_plot_values[:, 0].max() + .5
    range_min, range_max = scatter_plot_values[:, 1].min() - .5, scatter_plot_values[:, 1].max() + .5
    h = 0.01
    y_values, x_values = np.meshgrid(np.arange(domain_min, domain_max, h), np.arange(range_min, range_max, h))

    plot = pred_func(np.c_[y_values.ravel(), x_values.ravel()])

    plot = plot.reshape(y_values.shape)

    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.figure(facecolor="white")

    plt.ylim(range_min, range_max)
    plt.xlim(domain_min, domain_max)




    plt.contourf(y_values, x_values, plot, cmap=m)

    plt.scatter(scatter_plot_values[:,0], scatter_plot_values[:,1], s=30, c=class_membership, cmap = m)

    plt.show()
    return

def main():
    scatter_plot_values, class_membership  = sklearn.datasets.make_moons(200, noise=0.20)
    # Uncomment this chunk and then comment out the section below if you want to look through the available color schemes
    # They are all garbage.........Along with the way the graph looks in general
    # for m in plt.cm.datad:
    #         LR_function = sklearn.linear_model.LogisticRegressionCV()
    #         LR_function.fit(scatter_plot_values, class_membership)
    #         draw_boundary(lambda x: LR_function.predict(x),scatter_plot_values,class_membership,m) #,tableau20
    #         x = raw_input("\n\n Continue? (y/n): ")
    #         if x == 'n':
    #             break
    LR_function = sklearn.linear_model.LogisticRegressionCV()
    LR_function.fit(scatter_plot_values, class_membership)
    draw_boundary(lambda x: LR_function.predict(x),scatter_plot_values,class_membership) #,tableau20
    plt.title("Regression Prediction")

    network = Network(len(scatter_plot_values),0,3)
    quad =  network.train(scatter_plot_values,20000,class_membership)

    draw_boundary(lambda x: prediction_function(quad,x),scatter_plot_values,class_membership,tableau20)
    plt.title("Network Prediction")


main()








