Scorch
======
Scorch is a minimalist neural net framework in Scala inspired by PyTorch.


```scala
import scorch.autograd.Variable
import scorch.nn._
import botkop.{numsca => ns}

val numSamples = 128
val numClasses = 10
val nf1 = 40
val nf2 = 20

// Define the neural network
case class Net() extends Module {
  val fc1 = Linear(nf1, nf2) // an affine operation: y = Wx + b
  val fc2 = Linear(nf2, numClasses) // another one
  
  // glue the layers with a relu non-linearity: fc1 -> relu -> fc2
  override def forward(x: Variable): Variable = fc2(nn.relu(fc1(x)))
  
  // register the submodules to allow the world know what this net is composed of
  override def subModules(): Seq[Module] = Seq(fc1, fc2)
}

// instantiate
val net = Net()

// create an optimizer for updating the parameters
val optimizer = SGD(net.parameters(), lr = 0.01)

// random target and input to train on
val target = Variable(ns.randint(numClasses, Array(numSamples, 1)))
val input = Variable(ns.randn(numSamples, nf1))

for (j <- 0 to 1000) {

  // reset the gradients of the parameters
  optimizer.zeroGrad()

  // forward input through the network
  val output = net(input)
  
  // calculate the loss
  val loss = nn.softmax(output, target)

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
