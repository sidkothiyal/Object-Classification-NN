import layers
import feature

class NeuralNetwork:
    learning_rate = 0.3
    hidden_layers = []
    output_layer = None 
    input_layer = None
    def __init__(self, training_folder='train/', hidden_layer_sizes):
        self.training_folder = training_folder
        self.hidden_layer_sizes = hidden_layer_sizes

    def get_expected_output(output):
    	print output
    	expected_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    	inps = ['frog', 'bird', 'airplane', 'dog', 'deer', 'truck', 'automobile', 'horse', 'cat', 'ship']
    	for i, inp in enumerate(inps):
    		if inp == output:
    			expected_vector[i] += 1 
    	return expected_vector

    def train(self):
		for f in os.listdir(self.training_folder):
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

                self.input_layer.put_values(feature_vector)
                for hidden_layer in self.hidden_layers:
                    hidden_layer.calc_neuron_vals()
                self.output_layer.calc_neuron_vals()
                self.output_layer.put_values(expected_output)

                
	
    def  setup_architecture(self):
        expected_output = []
        feature_vector = []
        for f in os.listdir(self.training_folder):
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
        self.input_layer = layers.InputLayer(len(feature_vector))
        self.get_hidden_layers(self.hidden_layer_sizes)
        self.output_layer = layers.OutputLayer(len(expected_output))
        self.input_layer.setup_architecture()
        for i, hidden_layer in enumerate(self.hidden_layers):
            if i == 0:
                hidden_layer.setup_architecture(self.input_layer)
            else:
                hidden_layer.setup_architecture(self.hidden_layers[i-1])
        self.output_layer.setup_architecture(self.hidden_layers[-1])                

    def get_hidden_layers(self, hidden_layer_sizes):
        for hidden_layer_size in hidden_layer_sizes:
            self.hidden_layers.append(layers.HiddenLayer(hidden_layer_size))