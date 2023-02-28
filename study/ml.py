import numpy as np

class FCLayer():
	def __init__(self, intput_size, output_size):
		# out = W * in
		self.weight = np.random.rand(intput_size, output_size)
		self.bias = np.random.rand(output_size)

	def forward(self, ipt):
		# Assume input is N row vectors, N is batch size
		self.input = ipt
		output = np.dot(self.input, self.weight) + self.bias
		return output

	def backward(self, grad_out, step_size=1e-3):
		# assume grad_out is N x out, in is N x in, weight is in x out
		# out = W * in, so d(loss)/d(W) = d(loss)/d(out) * d(out)/d(weight)
		grad_weight = np.dot(self.input.T, grad_out)
		grad_bias = np.sum(grad_out, axis=0)
		self.weight -= step_size * grad_weight
		self.bias -= step_size * grad_bias
		# d(loss)/d(in) = d(loss)/d(out) * d(out)/d(in)
		return np.dot(grad_out, self.weight.T)

class BatchNorm():
	def __init__(self, D, is_train=True, momentum=0.9, epislon=1e-6):
		self.gamma = np.random.rand(D)
		self.beta = np.random.rand(D)
		self.momentum = momentum
		self.running_mean = 0.0
		self.running_var = 1.0
		self.epislon = epislon
		self.is_train = is_train

	def forward(self, ipt):
		N, _ = ipt.shape
		if self.is_train:
			var = np.var(ipt, axis=0, keepdims=False)
			if N == 1:
				var = np.maximum(1.0, var)
			mean = np.mean(ipt, axis=0, keepdims=False)
			self.running_mean = self.running_mean*self.momentum + mean * (1-self.momentum)
			self.running_var = self.running_var*self.momentum + var * (1-self.momentum)
		else:
			var = self.running_var
			mean = self.running_mean
		self.ipt = ipt
		self.mean = mean
		self.var = var + self.epislon
		self.std = np.sqrt(var + self.epislon)
		self.ipt_norm = (ipt-mean) / self.std
		opt = self.gamma * self.ipt_norm + self.beta
		return opt

	def backward(self, grad_out, step_size=1e-3):
		N, _ = grad_out.shape
		grad_gamma = np.sum(grad_out * self.ipt_norm, axis=0)
		grad_beta = np.sum(grad_out, axis=0)
		self.gamma -= step_size * grad_gamma
		self.beta -= step_size * grad_beta
		# d(loss)/d(ipt) = d(loss)/d(out) * d(out)/d(ipt_norm) * d(ipt_norm)/d(ipt)
		# d(ipt_norm)/d(ipt) = (d(de_mean)/d(ipt) * std + d(std)/d(ipt) * de_mean) / (std^2)
		#                    = ((1 - 1/N) * std + 0.5 * (2 ipt - 2 mean)^(-0.5) * de_mean)
		grad_ipt_norm = grad_out * self.gamma # N x D
		grad_x_center = grad_ipt_norm / self.std
		d_mean_d_ipt = 1/N
		d_ipt_center_d_ipt = 1 - d_mean_d_ipt
		d_std_d_var = 0.5 * self.var**(-0.5)
		d_var_d_ipt_center = 2 * (self.ipt - self.mean)
		d_std_d_ipt = d_std_d_var * d_var_d_ipt_center * d_ipt_center_d_ipt
		d_ipt_norm_d_ipt = (d_ipt_center_d_ipt * self.std + d_std_d_ipt * (self.ipt - self.mean)) / (self.std**2)
		grad_ipt = grad_ipt_norm * d_ipt_norm_d_ipt
		return grad_ipt

class Sigmoid():
	def __init__(self):
		pass

	def forward(self, ipt):
		opt = 1 / (1 + np.exp(-ipt))
		self.opt = opt
		return opt

	def backward(self, grad_opt):
		grad_ipt = grad_opt * self.opt * (1-self.opt)
		return grad_ipt

class MSELoss():
	def __init__(self):
		pass

	def forward(self, pred, gt):
		self.pred = pred
		self.gt = gt
		self.value = np.mean((pred-gt)**2, axis=(0,1))
		return self.value

	def backward(self):
		# loss = (in - gt)^2 = in^2 - 2in*gt + gt^2, d(loss)/d(in) = 2in - 2gt
		return 2 * (self.pred - self.gt)

class SequentialModel():
	def __init__(self):
		self.layers = []
		self.loss = None

	def add_layer(self, layer):
		self.layers.append(layer)

	def set_loss(self, loss):
		self.loss = loss

	def forward(self, ipt):
		for i in range(len(self.layers)):
			ipt = self.layers[i].forward(ipt)
		self.opt = ipt
		return self.opt

	def backward(self, gt):
		loss = self.loss.forward(self.opt, gt)
		grad = self.loss.backward()
		for i in range(len(self.layers)-1, -1, -1):
			grad = self.layers[i].backward(grad)
		return loss


def TestModel():
	batch_size = 4
	input_size = 3
	output_size = 2
	in_val = np.random.rand(batch_size, input_size)
	label = np.asarray([[1.0] * output_size] * batch_size)
	model = SequentialModel()
	model.add_layer(FCLayer(3, 2))
	model.add_layer(BatchNorm(2))
	model.add_layer(Sigmoid())
	model.set_loss(MSELoss())
	iteration = 0
	while True:
		print("iteration:", iteration)
		pred = model.forward(in_val)
		loss = model.backward(label)
		print(pred, loss)
		if loss < 1e-3:
			print("Converged at loss =", loss)
			break
		iteration += 1

if __name__ == '__main__':
	TestModel()

