import layers
import feature

class NeuralNetwork:
    learning_rate = 0.3

    def __init__(self, training_folder, hidden_layer_sizes, expected_output_class, classifying_db=''):
        self.classify = self.get_classification(expected_output_class)

    def get_expected_output(output):
    	print output
    	expected_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    	inps = ['frog', 'bird', 'airplane', 'dog', 'deer', 'truck', 'automobile', 'horse', 'cat', 'ship']
    	for i, inp in enumerate(inps):
    		if inp == output:
    			expected_vector[i] += 1 
    	return expected_vector

    def train(self):
		for f in os.listdir('train/'):
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
				

    def get_hidden_layers(self, hidden_layer_sizes):
        hl = []
        for hidden_layer_size in hidden_layer_sizes:
            hl.append(layers.HiddenLayer(hidden_layer_size))

        return []
