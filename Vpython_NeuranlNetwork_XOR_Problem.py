from vpython import *
import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []
        self.max_weight_change = 10.0
        self.weight_range = 2 * self.max_weight_change
        for i in range(len(layers)-1):
            self.weights.append(np.random.rand(layers[i], layers[i+1]) * self.weight_range - self.max_weight_change)
            self.biases.append(np.random.rand(1, layers[i+1]) * self.weight_range - self.max_weight_change)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feedforward(self, X):
        self.a = [X]
        for i in range(len(self.layers)-1):
            output = self.sigmoid(np.dot(self.a[i], self.weights[i]) + self.biases[i])
            self.a.append(output)
        return output

    def backpropagation(self, X, Y, output, learning_rate):
        self.error = Y - output
        d_weights = []
        d_biases = []
        for i in reversed(range(len(self.layers)-1)):
            d_weights.insert(0, self.a[i].T.dot(self.error * self.sigmoid_derivative(self.a[i+1])) * learning_rate)
            d_biases.insert(0, np.sum(self.error * self.sigmoid_derivative(self.a[i+1]), axis=0) * learning_rate)
            self.error = self.error.dot(self.weights[i].T)
        for i in range(len(self.layers)-1):
            self.weights[i] += d_weights[i]
            self.biases[i] += d_biases[i]

    def train(self, X, Y, learning_rate, epochs):
        for _ in range(epochs):
            output = self.feedforward(X)
            self.backpropagation(X, Y, output, learning_rate)

            # Update labels and cylinder colors in training process
            for i in range(len(connections)):
                for j in range(len(connections[i])):
                    for k in range(len(connections[i][j])):
                        if abs(self.weights[i][j][k]) > 10 : 
                            connections[i][j][k].radius = 0.05
                        else : 
                            connections[i][j][k].radius = self.weights[i][j][k]/200
                        
                        if self.weights[i][j][k] > 3:
                            connections[i][j][k].color = vec(self.weights[i][j][k]/20,0,0)
                        elif self.weights[i][j][k] < -3:
                            connections[i][j][k].color = vec(0,0,abs(self.weights[i][j][k])/20)
                        else : 
                            connections[i][j][k].color = color.white
                        weight_labels[i][j][k].text = f"w: {self.weights[i][j][k]:.2f}"
                        bias_labels[i][j].text = f"b: {self.biases[i][0][k]:.2f}"
            loss_label.text = f"Loss: {np.mean(np.square(Y - self.feedforward(X))):.4f},\n Y:{Y}, predict:{self.feedforward(X)}"
            rate(1000)

    def predict(self, X):
        return self.feedforward(X)

# Create neural network
nn = NeuralNetwork([2, 4, 1])

# Create vpython canvas
scene = canvas(title='Neural Network Visualization', width=800, height=400)
layers_x = [0, 1, 2, 3]
neurons = []
connections = []
weight_labels = []
bias_labels = []

for i, n in enumerate(nn.layers):
    layer = []
    for j in range(n):
        neuron = sphere(pos=vector(layers_x[i], j-n/2, 0), radius=0.1)
        layer.append(neuron)
    neurons.append(layer)

for i in range(len(neurons)-1):
    layer_connections = []
    layer_weight_labels = []
    layer_bias_labels = []
    for neuron1 in neurons[i]:
        neuron_connections = []
        neuron_weight_labels = []
        for neuron2 in neurons[i+1]:
            connection = cylinder(pos=neuron1.pos, axis=neuron2.pos-neuron1.pos, radius=0.01)
            neuron_connections.append(connection)
            weight_label_pos = connection.pos + (connection.axis / 2) + vec(0,(neuron1.pos.y)*0.2,0)
            weight_label = label(pos=weight_label_pos, text="")
            neuron_weight_labels.append(weight_label)
        layer_connections.append(neuron_connections)
        layer_weight_labels.append(neuron_weight_labels)
        bias_label = label(pos=neuron1.pos, text="")
        layer_bias_labels.append(bias_label)
    connections.append(layer_connections)
    weight_labels.append(layer_weight_labels)
    bias_labels.append(layer_bias_labels)

# Initialize training data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

# Create label to display loss
loss_label = label(pos=vector(3.2, 1.2, 0), text="")

# Train neural network
nn.train(X, Y, 0.1, 10000)
