import math

def output_function(x):
	return 1/(1 + math.exp(-x))


def output_summation(layer, i):
	sum = 0
	for neuron in layer.neurons:
		sum += neuron.get_summation(i)
	return sum

class HiddenLayer:
	bias = 0
	neurons = []
	prevLayer = InputLayer()
	def __init__(self, hidden_layer_size):
		super(HiddenLayer, self).__init__()
		self.size = hidden_layer_size

	def calc_neuron_vals(self):
		for i, neuron in enumerate(self.neurons):
			neuron.value = output_function(output_summation(self.prevLayer, i) + bias)

	def set_architecture(self, inputLayer):
		self.prevLayer = inputLayer
		bias = 0.5
		for i in xrange(self.size):
			self.neurons.append(Neuron)
			for j, neuron in enumerate(self.prevLayer.neurons):
				neuron.outgoing_weights.append(0)

class InputLayer:
	neurons = []
	def __init__(self, size):
		super(InputLayer, self).__init__()
		self.size = size

	def set_architecture(self):
		for i in xrange(self.size):
			self.neurons.append(Neuron())

	def put_values(self, feature_vector):
		for i, feature in enumerate(feature_vector):
			self.neurons[i].value = feature


class OutputLayer:
	bias = 0
	neurons = []
	prevLayer = HiddenLayer(0)
	def __init__(self, output_size):
		super(OutputLayer, self).__init__()
		self.size = output_size
	def calc_neuron_vals(self):
		for i, neuron in enumerate(self/neurons):
			neuron.value = output_function(output_summation(self.prevLayer, i) + bias)

	def set_architecture(self, hiddenLayer):
		self.prevLayer = hiddenLayer
		bias = 0.5
		for i in xrange(self.size):
			self.neurons.append(Neuron)
			for j, neuron in enumerate(self.prevLayer.neurons):
				neuron.outgoing_weights.append(0)


class Neuron:
	value = 0
	outgoing_weights = []
	def __init__(self):
		super(Neuron, self).__init__()

	def get_summation(self, i):
		return value * outgoing_weights[i]