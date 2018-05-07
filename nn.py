import numpy
import matplotlib.pyplot as plt
import sys
import scipy.special
import scipy.ndimage



class neuralnetwork:
	def __init__(self,input_nodes,hidden_nodes,output_nodes,learning_rate):
		self.inodes = input_nodes
		self.hnodes = hidden_nodes
		self.onodes = output_nodes

		self.lr = learning_rate

		self.wih = (numpy.random.rand(self.hnodes, self.inodes))-0.5

		# print (self.inodes)

		# print(self.wih.shape)
		self.who = (numpy.random.rand(self.onodes, self.hnodes))-0.5
		
		self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
		self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))


		self.activation_function = lambda x:  scipy.special.expit(x)
		self.reverse_activation_function = lambda x:  scipy.special.logit(x)


		pass

	def train(self, inputs, targets):

		inputs = numpy.array(inputs, ndmin=2).T
		targets = numpy.array(targets, ndmin=2).T

		# print(inputs.shape)
		# print(self.wih.shape)
		#calculate signals into hidden layer
		hidden_inputs = numpy.dot(self.wih, inputs) 
		#calculate the signals emerging from hidden layer 
		hidden_outputs = self.activation_function(hidden_inputs) 
		#calculate signals into final output layer

		final_inputs = numpy.dot(self.who, hidden_outputs) 
		#calculate the signals emerging from final output layer 
		final_outputs = self.activation_function(final_inputs)
		
		#error in the process
		output_errors = targets - final_outputs

		hidden_errors = numpy.dot(self.who.T, output_errors)

		#training process

		self.who += self.lr*numpy.dot((output_errors)*final_outputs*(1.0 - final_outputs),numpy.transpose(hidden_outputs))


		# print (inputs.shape)
		# print (numpy.dot((hidden_errors)*hidden_outputs*(1.0 - hidden_outputs),inputs).shape)
		self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))



		pass

	def query(self, inputs):
		#calculate signals into hidden layer 
		hidden_inputs = numpy.dot(self.wih, inputs) 
		#calculate the signals emerging from hidden layer 
		hidden_outputs = self.activation_function(hidden_inputs) 
		#calculate signals into final output layer 
		final_inputs = numpy.dot(self.who, hidden_outputs) 
		#calculate the signals emerging from final output layer 
		final_outputs = self.activation_function(final_inputs)
		
		return final_outputs
		pass

	def reverseQuery(self, targets):
		targets = numpy.array(targets, ndmin=2).T
		# print(self.wih.T.shape)
		# print(targets.shape)
		final_inputs = self.reverse_activation_function(targets)
		hidden_outputs = numpy.dot(self.who.T, final_inputs)
		
		hidden_outputs -= numpy.min(hidden_outputs)
		hidden_outputs /= numpy.max(hidden_outputs)
		hidden_outputs *= 0.98
		hidden_outputs += 0.01

		hidden_inputs = self.reverse_activation_function(hidden_outputs)

		input_layer = numpy.dot(self.wih.T, hidden_inputs)

		input_layer -= numpy.min(input_layer)
		input_layer /= numpy.max(input_layer)
		input_layer *= 0.98
		input_layer += 0.01


		return input_layer

		pass

	def reverseQueryImage(self, targets):
		inputs = self.reverseQuery(targets)

		# inputs = (inputs-0.01)*255/0.99
		image_array = numpy.asfarray(inputs).reshape((28,28))
		
		plt.imshow(image_array, cmap='Greys', interpolation='None')		
		plt.show()
		pass
