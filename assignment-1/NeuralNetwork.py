def learn(self, training_data):
	random.shuffle(training_data)
	for x, y in training_data:
		delta_biases, delta_weights = error(x, y)
		self.biases = [b - self.lr*d_b for b, d_b in zip(self.biases, delta_biases)]
		self.weights = [w - self.lr*d_w for w, d_w in zip(self.weights, delta_weights)] 	

def error(self, x, y):
	error_biases = [np.zeros(bias.shape) for bias in self.biases]
	error_weights = [np.zeros(weight.shape) for weight in self.weights]
	error_biases[-1] = (self.activations[-1] - y)*activation_function_grad(self.z[-1])
	error_weights[-1] = self.activations[-2]*np.transpose(error_biases[-1]) 
	for i in xrange(num_layers-2,1,-1):
		error = np.multiply(np.dot(error_weights[i+1], error_biases[i+1]), activation_function_grad(self.z[i]))
		error_biases[i] = error
		error_weights[i] = self.act[i-1]*np.transpose(error)
	return error_biases, error_weights
  	
