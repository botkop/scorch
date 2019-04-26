"What I cannot create, I do not understand." - Richard Feynman.


Scorch is a project that I wrote with the prime purpose of teaching myself the architecture of deep learning frameworks. I took PyTorch as a source of inspiration, because it has a nice imperative programming interface. It's written in Scala, and is not as performant as the big players. In order to achieve that, I would have to rely on existing C++ implementations of the algorithms, which would result in missing it's educational purpose. That said, it does contain most well known building blocks of neural networks, and since it's written for the JVM, it can leverage the features of this ecosystem. As an example of the latter, see my project [Akkordeon: Training neural networks with Akka.](https://github.com/botkop/akkordeon)


Scorch
======

[![Join the chat at https://gitter.im/botkop/scorch](https://badges.gitter.im/botkop/scorch.svg)](https://gitter.im/botkop/scorch?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Scorch is a deep learning framework in Scala inspired by PyTorch.

It has [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) built in 
and follows an [imperative coding style](https://mxnet.incubator.apache.org/architecture/program_model.html#symbolic-vs-imperative-programs).

Scorch uses [numsca](https://github.com/botkop/numsca) for creation and processing of Tensors.

Here's an example of a convolutional neural net, with relu and pooling followed by 2 affine layers:

```scala
package scorch.sandbox

import botkop.{numsca => ns}
import scorch._
import scorch.autograd.Variable
import scorch.nn.cnn._
import scorch.nn._
import scorch.optim.SGD

object ReadmeConvNet extends App {

  // input layer shape
  val (numSamples, numChannels, imageSize) = (8, 3, 32)
  val inputShape = List(numSamples, numChannels, imageSize, imageSize)

  // output layer
  val numClasses = 10

  // network blueprint for conv -> relu -> pool -> affine -> affine
  case class ConvReluPoolAffineNetwork() extends Module {

    // convolutional layer
    val conv = Conv2d(numChannels = 3, numFilters = 32, filterSize = 7, weightScale = 1e-3, pad = 1, stride = 1)
    // pooling layer
    val pool = MaxPool2d(poolSize = 2, stride = 2)

    // calculate number of flat features
    val poolOutShape = pool.outputShape(conv.outputShape(inputShape))
    val numFlatFeatures = poolOutShape.tail.product // all dimensions except the batch dimension

    // reshape from 3d pooling output to 2d affine input
    def flatten(v: Variable): Variable = v.reshape(-1, numFlatFeatures)

    // first affine layer
    val fc1 = Linear(numFlatFeatures, 100)
    // second affine layer (output)
    val fc2 = Linear(100, numClasses)

    // chain the layers in a forward pass definition
    override def forward(x: Variable): Variable =
      x ~> conv ~> relu ~> pool ~> flatten ~> fc1 ~> fc2
  }

  // instantiate the network, and parallelize it
  val net = ConvReluPoolAffineNetwork().par()

  // stochastic gradient descent optimizer for updating the parameters
  val optimizer = SGD(net.parameters, lr = 0.001)

  // random input and target
  val input = Variable(ns.randn(inputShape: _*))
  val target = Variable(ns.randint(numClasses, Array(numSamples, 1)))

  // loop (should reach 100% accuracy in 2 steps)
  for (j <- 0 to 3) {

    // reset gradients
    optimizer.zeroGrad()

    // forward pass
    val output = net(input)

    // calculate the loss
    val loss = softmaxLoss(output, target)

    // log accuracy
    val guessed = ns.argmax(output.data, axis = 1)
    val accuracy = ns.sum(target.data == guessed) / numSamples
    println(s"$j: loss: ${loss.data.squeeze()} accuracy: $accuracy")

    // backward pass
    loss.backward()

    // update parameters with gradients
    optimizer.step()
  }
}
```

The documentation below is a copy of the Autograd and Neural Networks sections of 
[PyTorch blitz](http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html),
adapted for Scorch.

## Automatic differentiation

Central to all neural networks in Scorch is the autograd package. 
Let’s first briefly visit this, and we will then go to training our first neural network.

The `autograd` package provides automatic differentiation for all operations on Tensors. 
It is a define-by-run framework, which means that your backprop is defined by how your code is run, and that every single iteration can be different.

### Variable
`autograd.Variable` is the central class of the package. 
It wraps a [numsca](https://github.com/botkop/numsca) `Tensor`, and supports nearly all the operations defined on it. 
Once you finish your computation you can call `.backward()` and have all the gradients computed automatically.

You can access the raw tensor through the `.data` attribute, while the gradient w.r.t. this variable is accumulated into `.grad`.

### Function
There’s one more class which is very important for autograd implementation - a `Function`.

`Variable` and `Function` are interconnected and build up an acyclic graph, 
that encodes a complete history of computation. 
Each variable has a `.gradFn` attribute that references the `Function` that has created the `Variable` 
(except for Variables created by the user - their `gradFn` is `None`).

If you want to compute the derivatives, you can call `.backward()` on a `Variable`. 
If you do not specify a gradient argument, then Scorch will create one for you on the fly,
of the same shape as the Variable, and filled with all ones. 
(This is different from Pytorch)

```scala
import scorch.autograd.Variable
import botkop.{numsca => ns}
```
Create a Variable:
```scala
val x = Variable(ns.ones(2,2))
```
```text
x: scorch.autograd.Variable =
data: [[1.00,  1.00],
 [1.00,  1.00]]
```
Do an operation on the Variable:
```scala
val y = x + 2
```
```
y: scorch.autograd.Variable =
data: [[3.00,  3.00],
 [3.00,  3.00]]
```
`y` was created as a result of an operation, so it has a `gradFn`.
```scala
println(y.gradFn)
```
```
Some(AddConstant(data: [[1.00,  1.00],
 [1.00,  1.00]],2.0))
```
Do more operations on `y`

```scala
val z = y * y * 3
val out = z.mean()
```
```text
z: scorch.autograd.Variable =
data: [[27.00,  27.00],
 [27.00,  27.00]]
out: scorch.autograd.Variable = data: 27.00
```
### Gradients
Let’s backprop now, and print gradients d(out)/dx.
```scala
out.backward()
println(x.grad)
```
```text
data: [[4.50,  4.50],
 [4.50,  4.50]]
```

## Neural Networks
Neural networks can be constructed using the `scorch.nn` package.

Now that you had a glimpse of `autograd`, `nn` depends on autograd to define models and differentiate them. 
An `nn.Module` contains layers, and a method `forward(input)` that returns the output.

A typical training procedure for a neural network is as follows:

* Define the neural network that has some learnable parameters (or weights)
* Iterate over a dataset of inputs
* Process input through the network
* Compute the loss (how far is the output from being correct)
* Propagate gradients back into the network’s parameters
* Update the weights of the network, typically using a simple update rule: 

  `weight = weight - learningRate * gradient`
  
### Define the network
Let’s define this network:
```scala
import scorch.autograd.Variable
import scorch.nn._
import scorch._

val numSamples = 128
val numClasses = 10
val nf1 = 40
val nf2 = 20

// Define a simple neural net
case class Net() extends Module {
  val fc1 = Linear(nf1, nf2) // an affine operation: y = Wx + b
  val fc2 = Linear(nf2, numClasses) // another one

  // glue the layers with a relu non-linearity: fc1 -> relu -> fc2
  override def forward(x: Variable): Variable = 
    x ~> fc1 ~> relu ~> fc2
}

val net = Net()
```
You just have to define the forward function.
The backward function (where gradients are computed) is automatically defined for you using autograd.

The learnable parameters of a model are returned by `net.parameters`

```scala
val params = net.parameters
println(params.length)
println(params.head.shape)
```
```text
4
List(20, 40)
```
The input to the forward method is an `autograd.Variable`, and so is the output. 
```scala
import botkop.{numsca => ns}

val input = Variable(ns.randn(numSamples, nf1))
val out = net(input)
println(out)
println(out.shape)
```
```text
data: [[1.60,  -0.22,  -0.66,  0.86,  -0.59,  -0.80,  -0.40,  -1.37,  -1.94,  1.23],
 [1.15,  -3.81,  5.45,  6.81,  -3.02,  2.35,  3.75,  1.79,  -7.31,  3.60],
 [3.12,  -0.94,  2.69, ...
 
List(128, 10)
```
Zero the gradient buffers of all parameters and backprop with random gradients.
```scala
net.zeroGrad()
out.backward(Variable(ns.randn(numSamples, numClasses)))
```

Before proceeding further, let’s recap all the classes you’ve seen so far.

__Recap:__
* `numsca.Tensor` - A multi-dimensional array.
* `autograd.Variable` - Wraps a Tensor and records the history of operations applied to it. 

  Has (almost) the same API as a `Tensor`, with some additions like `backward()`. Also holds the gradient w.r.t. the tensor.
  
* `nn.Module` - Neural network module. Convenient way of encapsulating parameters.
* `autograd.Function` - Implements forward and backward definitions of an autograd operation. 

  Every `Variable` operation, creates at least a single `Function` node, 
  that connects to functions that created a `Variable` and encodes its history.

__At this point, we covered:__

* Defining a neural network
* Processing inputs and calling backward

__Still Left:__

* Computing the loss
* Updating the weights of the network

### Loss function
A loss function takes the (output, target) pair of inputs, 
and computes a value that estimates how far away the output is from the target.

There are several different loss functions under the `scorch` package . 
A common loss is: `scorch.softmaxLoss` which computes the softmax loss between the input and the target.

For example:
```scala
val target = Variable(ns.randint(numClasses, Array(numSamples, 1)))
val output = net(input)
val loss = softmaxLoss(output, target)
println(loss)
```
```text
data: 5.61
```

Now, if you follow loss in the backward direction, 
using its `.gradFn` attribute, you will see a graph of computations that looks like this:
```text
input -> linear -> relu -> linear
      -> SoftmaxLoss
      -> loss
```
So, when we call `loss.backward()`, 
the whole graph is differentiated w.r.t. the loss, 
and all Variables in the graph will have their `.grad` Variable accumulated with the gradient.

### Backprop

To backpropagate the error all we have to do is to call `loss.backward()`. 
You need to clear the existing gradients though, else gradients will be accumulated to existing gradients.

Now we will call `loss.backward()`, and have a look at fc1's bias gradients before and after the backward.

```scala
net.zeroGrad()
println("fc1.bias.grad before backward")
println(fc1.bias.grad)
loss.backward()
println("fc1.bias.grad after backward")

```
```text
fc1.bias.grad before backward
data: [0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00]
fc1.bias.grad after backward
data: [0.07,  0.20,  0.21,  -0.04,  0.16,  0.09,  0.34,  -0.06,  0.17,  -0.06,  0.02,  -0.01,  -0.07,  0.09,  0.12,  -0.04,  0.19,  0.28,  0.06,  0.13]
```

Now, we have seen how to use loss functions.

__The only thing left to learn is:__

* Updating the weights of the network

### Update the weights

The simplest update rule used in practice is the Stochastic Gradient Descent (SGD):

  `weight = weight - learningRate * gradient`
  
We can implement this using simple scala code:

```scala
net.parameters.foreach(p => p.data -= p.grad.data * learningRate)
```

However, as you use neural networks, you want to use various different update rules such as 
SGD, Nesterov, Adam, etc. 
To enable this, we built a small package: scorch.optim that implements these methods. Using it is very simple:
```scala
import scorch.optim.SGD

// create an optimizer for updating the parameters
val optimizer = SGD(net.parameters, lr = 0.01)

// in the training loop:

optimizer.zeroGrad()                   // reset the gradients of the parameters
val output = net(input)                // forward input through the network
val loss = softmaxLoss(output, target) // calculate the loss
loss.backward()                        // back propagate the derivatives
optimizer.step()                       // update the parameters with the gradients
```

## Wrap up
To wrap up, here is a complete example:

```scala
import botkop.{numsca => ns}
import scorch.autograd.Variable
import scorch.optim.SGD
import scorch._

val numSamples = 128
val numClasses = 10
val nf1 = 40
val nf2 = 20

// Define a simple neural net
case class Net() extends Module {
  val fc1 = Linear(nf1, nf2) // an affine operation: y = Wx + b
  val fc2 = Linear(nf2, numClasses) // another one

  // glue the layers with a relu non-linearity: fc1 -> relu -> fc2
  override def forward(x: Variable) = x ~> fc1 ~> relu ~> fc2
}

// instantiate
val net = Net()

// create an optimizer for updating the parameters
val optimizer = SGD(net.parameters, lr = 0.01)

// random target and input to train on
val target = Variable(ns.randint(numClasses, Array(numSamples, 1)))
val input = Variable(ns.randn(numSamples, nf1))

for (j <- 0 to 1000) {

  // reset the gradients of the parameters
  optimizer.zeroGrad()

  // forward input through the network
  val output = net(input)

  // calculate the loss
  val loss = softmaxLoss(output, target)

  // print loss and accuracy
  if (j % 100 == 0) {
    val guessed = ns.argmax(output.data, axis = 1)
    val accuracy = ns.sum(target.data == guessed) / numSamples
    println(s"$j: loss: ${loss.data.squeeze()} accuracy: $accuracy")
  }

  // back propagate the derivatives
  loss.backward()

  // update the parameters with the gradients
  optimizer.step()
}
```

## Contributors
Thanks to [Jasper](https://github.com/Jasper-M) for helping out with Scala type inference magic far beyond my capabilities.

## Dependency
Add this to build.sbt:
```scala
libraryDependencies += "be.botkop" %% "scorch" % "0.1.0"
```

## References
- [Deep Learning with PyTorch: A 60 Minute Blitz](http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- [Backpropagation, Intuitions](http://cs231n.github.io/optimization-2/)
- [Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/pdf/1502.05767.pdf)
- [Automatic differentiation](http://www.pvv.ntnu.no/~berland/resources/autodiff-triallecture.pdf)
- [Derivative Calculator with step-by-step Explanations](http://calculus-calculator.com/derivative/)
- [Differentiation rules](https://en.wikipedia.org/wiki/Differentiation_rules)
