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
    val conv = Conv2d(numChannels = 3, numFilters = 32, filterSize = 7, weightScale = 1e-3, stride = 1, pad = 1)
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
