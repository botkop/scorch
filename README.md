Scorch
======
Scorch is a minimalistic neural net framework in Scala inspired by PyTorch.


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
  override def forward(x: Variable): Variable = fc2(nn.relu(fc1(x)))
  override def subModules(): Seq[Module] = Seq(fc1, fc2)
}

val net = Net()
val optimizer = SGD(net.parameters(), lr = 0.01)

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
  
  // update the parameters
  optimizer.step()
}
```