Scorch
======
Scorch is a lightweight neural net framework in Scala inspired by PyTorch.

It has [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) built in and follows an [imperative coding style](https://mxnet.incubator.apache.org/architecture/program_model.html#symbolic-vs-imperative-programs).

# Automatic differentiation

Central to all neural networks in Scorch is the autograd package. 
Let’s first briefly visit this, and we will then go to training our first neural network.

The `autograd` package provides automatic differentiation for all operations on Tensors. 
It is a define-by-run framework, which means that your backprop is defined by how your code is run, and that every single iteration can be different.

## Variable
`autograd.Variable` is the central class of the package. 
It wraps a [numsca](https://github.com/botkop/numsca)`Tensor`, and supports nearly all the operations defined on it. 
Once you finish your computation you can call `.backward()` and have all the gradients computed automatically.

You can access the raw tensor through the `.data` attribute, while the gradient w.r.t. this variable is accumulated into `.grad`.

## Function
There’s one more class which is very important for autograd implementation - a `Function`.

`Variable` and `Function` are interconnected and build up an acyclic graph, 
that encodes a complete history of computation. 
Each variable has a `.gradFn` attribute that references the `Function` that has created the `Variable` 
(except for Variables created by the user - their `gradFn` is `None`).

If you want to compute the derivatives, you can call `.backward()` on a `Variable`. 
If you do not specify a gradient argument, then Scorch will create one for you on the fly,
of the same shape as the Variable, and filled with ones. 
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
## Gradients
Let’s backprop now, and print gradients d(out)/dx.
```scala
out.backward()
println(x.grad)
```
```text
data: [[4.50,  4.50],
 [4.50,  4.50]]
```

# Neural Networks
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

  `weight = weight - learning_rate * gradient`
  
## Define the network
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
  override def forward(x: Variable) = fc2(relu(fc1(x)))

  // register the submodules to allow the world to know what this net is composed of
  override def subModules = Seq(fc1, fc2)
}

val net = Net()
```
You just have to define the forward function, and list the layers as the output of the subModules method.
The backward function (where gradients are computed) is automatically defined for you using autograd.


------
Example:

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
  override def forward(x: Variable) = fc2(relu(fc1(x)))

  // register the submodules to allow the world to know what this net is composed of
  override def subModules = Seq(fc1, fc2)
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

## Dependency
Add this to build.sbt:
```scala
libraryDependencies += "be.botkop" %% "scorch" % "0.1.0-SNAPSHOT"
```

Scorch uses [numsca](https://github.com/botkop/numsca)

## References
- [Deep Learning with PyTorch: A 60 Minute Blitz](http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- [Backpropagation, Intuitions](http://cs231n.github.io/optimization-2/)
- [Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/pdf/1502.05767.pdf)
- [Automatic differentiation](http://www.pvv.ntnu.no/~berland/resources/autodiff-triallecture.pdf)
- [Derivative Calculator with step-by-step Explanations](http://calculus-calculator.com/derivative/)
- [Differentiation rules](https://en.wikipedia.org/wiki/Differentiation_rules)
