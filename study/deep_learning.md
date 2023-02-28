# Deep Learning

## Graph Neural Network

<br>

## Batch Normalization
Batch normalization is usually over channel. Which means we learn 2 scalars for each channel. This requires us to find mean and std over all pixels acroos the whole batch, for each channel.
$$bn(x)=\gamma \frac{x-\mu}{\sigma + \epsilon} + \beta $$
Helpful for training very depp neural networks. It is helpful in a few ways
- Prevent covariate shift. When lower layers shift the distribution, the uppler layer needs to adjust to it. By Batch Norm, we restrict the shift of the distribution.
- The distribution change of lower layer will counpond. If it shifts too much, it might cause saturation in upper layers, to stop the training earlier. So it could speed up the training.

Why we again add $\gamma,\beta$ to the function? Becasue otherwise we'll restrict the expression power of the model.

We need to also keep a running mean and var during training, in order to use it during testing.

Usually we put batch norm before activation function. But there's no final conclusion on this.

There's also layer normalization (normalizae per input, over all channels and pixels) and instance normalization (normalize per input and per channel). They don't rely on big batch size. And they are more useful for GAN, because each input has its own style, we don't want to normalize it with other inputs.

<br>

## Regularization
Regularization could be used to prevent overfitting. It is preferred to use a big model with regularization than using a small model without regularization. But why can Regularization prevent overfitting? Because it drives all weights towards zero. A sign of overfitting is some large weights.
### L2 - Ridge regularization - Weight Decay
When performing gradient descent, the actuall effect of L2 regularization is equivalent to multiply the weights with a constant multiplier to shrink the weights, aka weight decay! 
### L1 - Lasso
Instead of multiply the weights with a constant, the effect of L1 regularization is subtracting a constant value that have the same sign as the weight. L1 regularization induces sparsity, so it can be used for feature selection.
### Early stopping
Yeah, it is also a regularization!
### Dropout
Yeah, it is also a regularization! Dropout could also be viewed as a bagging/ensembling. Another interesting view is dropout could also be viewed as a noise that is injected to the model during training.

<br>

## Attention
![](imgs/attention.png)
$$attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d}})V$$

### Scaled dot product attention
The reason for that scale by sqrt dimension d is, as dimension increase, the inner product's value will become larger and larger, the relative difference between large values and small values of the inner product will also be larger and larger. This is bad for learning with softmax because after softmax, the values will be extreme, close to 1 or 0. Then the gradients will be small. So dividing by  a constant can make the values less extreme.

### self vs cross attention
In self attention, K, V, Q are all generated from the same input sequence. In cross attention, K, V are from one sequence, and Q is from another sequence.

### Mulit-head attention
Multi-head attention is trying to mimic conv nets where multiple different features could be learned in paralle.

### Positional enciding
Because attention model does not have information about ordering of the inputs, so positional encoding is important.

### Layer normalizatoin
Attention models usually use layer norm instead of batch norm. One reason is when batching the inputs, each input sequence may not be having the same length. So it does not make sense to normalize across the batch.

<br>

## ADAM optimizer
It is a combination of momentum and RMSProp. Momentum is $\alpha G_{t-1} + (1-\alpha) G_t$ which trys to reduce oscillation during gradient descent. RMSProp adjusts the step size/learning rate for each parameter individually. Adam, except for keeping a runnin mean of the gradients with momentum, also keeps a running squared gradient with 2nd degree momentum. And it divides the gradient with the equare root of that ruuning square. This scales small gradients up and big gradients down. Its helpful to get us out from saddle points.

<br>

## Unsupervised, semi-supervised, self-supervised
### Self-supervised - a type of unsupervised learning
This line of work aims to learn image representations without requiring human-annotated labels and then use those learned representations on some downstream tasks. For example, we can divide the image into pieces and randomly perturb them, let the model to figure out the correct order. Then we use transfer learning to use this learned feature on a different task.
### Semi-supervised Learning
Semi-supervised learning aims to use both labeled and unlabeled data at the same time to improve the performance of a supervised model. An example of this is FixMatch paper where you train your model on labeled images. Then, for your unlabeled images, you apply augmentations to create two images for each unlabeled image. Now, we want to ensure that the model predicts the same label for both the augmentations of the unlabeled images. This can be incorporated into the loss as a cross-entropy loss.
### Unsupervised learning
The goal is to find underlying patterns with each dataset, such as clustering.

<br>

## ConveNets
### Replace large kernels with smaller kernels
It is also a kind of regularization, because it reduced the number of parameters.
Pros:
- Reduce number of parameter
- Bigger non-linearity

Cons:
- Deeper, so harder to train
### Depthwise convolution
Perform convolutions per channel, then stack the per channel outputs, and apply multiple 1x1 convolutions on the stacked features to form the final output. It reduces the number of parameters significantly.

<br>

## Gradient explosion & Vanishing gradient
Gradient explosion identify: Huge gradients, weights, expecially lower layers, change drastically, weights become NaN, avalanche learning.

Vanishing gradient identify: Small gradients, especially lower layers. Lower layer weights change slowly, model not feel like being trained.

Solutoins:
- proper weight init
- batch norm
- gradient clipping
- use non-saturating activation functions
- use skip connection

<br>

## Anchorless vs Anchor Based
Anchorless are like CenterNet, ExtremeNet etc. For every pixel, they directly predict a bbox that is relative to the entire image. They don't rely on anchors, so they are less limited to predict huge or tiny objects. We could also potentially avoid NMS with anchorless detectors. Say if we generate GT by using a gaussian to blur a center object pixel, then we could run a 3x3 max pool over the output, then compare the results before and after max pool. The place where value stays the same is a peak and hence an object center. For anchorless, it is tricky to associte pixels to GT if a pixel corresponds to multiple objects.

<br>

## 2 Stage vs 1 Stage detectors
2 stage detectors, such as faster-RCNN, propose the object proposals first, then classify and refine each proposal. The object proposal network is trained with a regression head and a objectness score head. The classification head, takes roi poooled features, is trained with a multi class classification head and a regression head.

1 stage detectors, such as YOLO, divide the image into a sparse grid (low res feature map) and predicts a set of scores for every pixel. The set of score is NUM_ANCHORS * (4 + num_classes + 1)

<br>

## How to choose learning rate or multi loss weights?

<br>

## Data drift detection / model monitoring

<br>

## Ensemble
Pros:
- reduce variance
- give better accuracy

How to get diff models?
- vary the data
  - hide data
  - random sampling with replacement
- vary the model
  - architechure
  - initialization
- vary the hyper parameter
  - Horizontal epoch
  - Vertical epoch

Ways:
- average / weighted average
- Learned classifier on top of model classifications
- Boosting, but it is complicated
- model weight averaging, average weights of same models

## YOLO v1 - v4
YOLOv1 1 box per grid

YOLOv2 added anchor box and batch norm

YOLOv3 used skip connection hence better backbone, and spatial pynramid prediction - it detects objects on 3 levels of feature maps

YOLOv4 Data augmentation such as mixup, better activation such as mish, better loss such as gIoU

## Why use convolution instead of FC layers?
- Convolution is faster and fewer params
- It leverages and preserves locality property of image feature
- Convolution plus max pooling is translation invariant, which is an important character of image data. But FCN will not be translational invariant.

## What are the steps to prepare data?
- Data cleaning
  - Missing labels
  - Missing features
- Data balancing
- Normalization
- Data transform
- Feature selection
- Dimension reduction

## Implement DL modules
- TODO: conv2d
```py
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
```

## GRU vs LSTM
- GRU performance is on par with LSTMs. But they are simpler, and computationally, it is more efficient.
- GRU also train faster, especially better on less training data.
- LSTM may remember longer sequences than GRUs.

## Mahalanobis distance
The Mahalanobis distance is a measure of distance between a point $P$ and a distribution $D$. It is a multi-dimensional generalization of the idea of measuring how many standard deviations away $P$ is from the mean of $D$. If the distribution is rescaled to have unit variance, then Mahalanobis distance is the same as Euclidean distance.

Given a probability distribution $Q$ with mean $y$, and covariance matrix $S$, the Mahalanobis distance of a point $x$ from $Q$ is:

$$d_M(x,y,Q)=\sqrt{(x-y)^T S^{-1} (x-y)^T}$$
