import math

#TODO bias and bias wt

def softmax(output_layer_vals):
	softmax_vals = []
	sum_of_vals = 0
	for i, output_layer_val in enumerate(output_layer_vals):
		sum_of_vals += math.exp(output_layer_val)
	
	for i, output_layer_val in enumerate(output_layer_vals):
		softmax_vals.append(math.exp(output_layer_val)/sum_of_vals)

	return softmax_vals	

def derivative_softmax(x):
	return x * (1-x)

def leaky_relu(x):
	if x < 0:
		return 0.01 *x
	else:
		return x	

def derivative_leaky_relu(x):
	if x < 0:
		return 0.001
	else:
		return 1
			
def output_function(x):
	return 1/(1 + math.exp(-x))

def output_summation(layer, i):
	sum = 0
	for neuron in layer.neurons:
		sum += neuron.get_summation(i)
	return sum


class HiddenLayer:
	bias = 1
	neurons = []
	prevLayer = InputLayer()
	nextLayer = HiddenLayer()
	inputSummation = []

	def __init__(self, hidden_layer_size):
		super(HiddenLayer, self).__init__()
		self.size = hidden_layer_size

	def calc_neuron_vals(self):
		self.inputSummation = []
		for i, neuron in enumerate(self.neurons):
			self.inputSummation.append(output_summation(self.prevLayer, i) + self.bias)
			neuron.value = leaky_relu(self.inputSummation[-1])

	def set_architecture(self, prevLayer, nextLayer):
		self.prevLayer = prevLayer
		self.nextLayer = nextLayer
		bias = 0.5
		for i in xrange(self.size):
			self.neurons.append(Neuron())
			for j, neuron in enumerate(self.prevLayer.neurons):
				neuron.outgoing_weights.append(0)

	def change_weights(self, rate):
		for i, neuron in enumerate(self.neurons):
			neuron.error = derivative_leaky_relu(self.inputSummation[i])
			sumNextErrors = 0
			for j, nextNeuron in enumerate(self.nextLayer.neurons):
				sumNextErrors += neuron.outgoing_weights[i] * nextNeuron.error
			neuron.error *= sumNextErrors
			for j, prevNeuron in enumerate(self.prevLayer.neurons):
				weightDiff = - rate * neuron.error * prevNeuron.value
				prevNeuron.outgoing_weights[i] += weightDiff	

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
	bias = 1
	neurons = []
	prevLayer = HiddenLayer(0)
	inputSummation = []

	def __init__(self, output_size):
		super(OutputLayer, self).__init__()
		self.size = output_size

	def calc_neuron_vals(self):
		self.inputSummation = []
		for i, neuron in enumerate(self.neurons):
			self.inputSummation.append(output_summation(self.prevLayer, i) + self.bias)
		neuron_vals = softmax(self.inputSummation)
		for i, neuron in enumerate(self.neurons):
			neuron.value = neuron_vals[i]

	def set_architecture(self, hiddenLayer):
		self.prevLayer = hiddenLayer
		bias = 0.5
		for i in xrange(self.size):
			self.neurons.append(Neuron())
			for j, neuron in enumerate(self.prevLayer.neurons):
				neuron.outgoing_weights.append(0)

	def put_values(self, expected_output):
		self.expected_output = expected_output

	def output_diff(self):
		diff = []
		for i in xrange(self.size):
			dif.append(self.expected_output[i] - self.neurons[i].value)
		return diff

	def change_weights(self, rate):
		diff = self.output_diff()
		for i, neuron in enumerate(self.neurons):
			neuron.error = diff[i] * derivative_softmax(self.inputSummation[i])
			for j, prevNeuron in enumerate(self.prevLayer.neurons):
				weightDiff = - rate * neuron.error * prevNeuron.value
				prevNeuron.outgoing_weights[i] += weightDiff


class Neuron:
	value = 0
	outgoing_weights = []
	error = 0
	def __init__(self):
		super(Neuron, self).__init__()

	def get_summation(self, i):
		return value * outgoing_weights[i]