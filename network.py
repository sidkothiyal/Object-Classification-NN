import layers
import feature
import os
import csv
import linecache

class NeuralNetwork:
    p = 0
    f = 0
    learning_rate = 0.3
    hidden_layers = []
    output_layer = None 
    input_layer = None
    def __init__(self, hidden_layer_sizes, training_folder='train/'):
        self.training_folder = training_folder
        self.hidden_layer_sizes = hidden_layer_sizes

    def get_expected_output(self, output):
        expected_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        inps = ['frog', 'bird', 'airplane', 'dog', 'deer', 'truck', 'automobile', 'horse', 'cat', 'ship']
        for i, inp in enumerate(inps):
            if inp == output:
                expected_vector[i] += 1 
        return expected_vector

    def train(self):
        count = 0
        for f in os.listdir(self.training_folder):
            count += 1
            if count%100 == 0:
                print count
            if '.png' in f:
                f = f.split('.')
                f = f[0]
                if int(f) < 30000:
                    feature_vector = feature.get_features('train/'+f+'.png')
                    theline = linecache.getline('trainLabels.csv', int(f) + 1)
                    theline = theline.strip(' ')
                    theline = theline.strip('\n')
                    theline = theline.split(',')
                    theline = theline[1]
                    expected_output = self.get_expected_output(theline)

                    self.input_layer.put_values(feature_vector)
                    for hidden_layer in self.hidden_layers:
                        hidden_layer.calc_neuron_vals()
                    self.output_layer.calc_neuron_vals()
                    self.output_layer.put_values(expected_output)
                    self.back_propagate()

    def back_propagate(self):
        self.output_layer.change_weights(self.learning_rate)
        for hidden_layer in reversed(self.hidden_layers):
            hidden_layer.change_weights(self.learning_rate)

    def setup_architecture(self):
        expected_output = []
        feature_vector = []
        count = 0
        for f in os.listdir(self.training_folder):
            if count > 0:
                break
            if '.png' in f:
                feature_vector = feature.get_features('train/'+f)
                f = f.split('.')
                f = f[0]
                theline = linecache.getline('trainLabels.csv', int(f) + 1)
                theline = theline.strip(' ')
                theline = theline.strip('\n')
                theline = theline.split(',')
                theline = theline[1]
                expected_output = self.get_expected_output(theline)
                count += 1

        self.input_layer = layers.InputLayer(len(feature_vector))
        self.get_hidden_layers(self.hidden_layer_sizes)
        self.output_layer = layers.OutputLayer(len(expected_output))
        if len(self.hidden_layer_sizes) == 0:
            self.input_layer.set_architecture(self.output_layer)
        else:
            self.input_layer.set_architecture(self.hidden_layers[0])    
        for i, hidden_layer in enumerate(self.hidden_layers):
            prevLayer = None
            nextLayer = None
            if i == 0:
                prevLayer = self.input_layer
            else:
                prevLayer = self.hidden_layers[i-1]
            if i == len(self.hidden_layers) - 1:
                nextLayer = self.output_layer
            else:
                nextLayer = self.hidden_layers[i+1]        
            hidden_layer.set_architecture(prevLayer, nextLayer)    
        self.output_layer.set_architecture(self.hidden_layers[-1])                

    def get_hidden_layers(self, hidden_layer_sizes):
        for hidden_layer_size in hidden_layer_sizes:
            self.hidden_layers.append(layers.HiddenLayer(hidden_layer_size))


    def test(self):
        for f in os.listdir(self.training_folder):
            if '.png' in f:
                f = f.split('.')
                f = f[0]
                if int(f) >= 30000:
                    feature_vector = feature.get_features('train/'+f + '.png')
                    theline = linecache.getline('trainLabels.csv', int(f) + 1)
                    theline = theline.strip(' ')
                    theline = theline.strip('\n')
                    theline = theline.split(',')
                    theline = theline[1]
                    expected_output = self.get_expected_output(theline)

                    self.input_layer.put_values(feature_vector)
                    for hidden_layer in self.hidden_layers:
                        hidden_layer.calc_neuron_vals()
                    self.output_layer.calc_neuron_vals()
                    self.output_layer.put_values(expected_output)
                    output = self.output_layer.get_output()
                    if output == theline:
                        self.p += 1
                    else:
                        self.f += 1    

if __name__ == '__main__':
    nn = NeuralNetwork(hidden_layer_sizes=[90, 60, 30])
    nn.setup_architecture()
    for i in xrange(10000):
        print i
        for neuron in nn.input_layer.neurons:
            print neuron.outgoing_weights
        for j in nn.hidden_layers:
            for neuron in j.neurons:
                print neuron.outgoing_weights
        for neuron in nn.output_layer.neurons:
            print neuron.outgoing_weights
        nn.train()
    nn.test()
    print "true ", nn.p
    print "false ", nn.f
    print "percentage ", nn.p/(nn.p + nn.f)
