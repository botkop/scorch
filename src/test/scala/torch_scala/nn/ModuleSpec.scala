package torch_scala.nn

import org.scalatest.{FlatSpec, Matchers}
import torch_scala.api.aten.{CPU, Shape, Tensor}
import torch_scala.autograd.{SoftmaxLoss, Variable}
import torch_scala.autograd.MathVariable._
import torch_scala.optim.{Adam, SGD}
import torch_scala.api.aten.functions.Math._

class ModuleSpec extends FlatSpec with Matchers {

  "A Module" should "compute a single pass through a linear network" in {

    val numSamples = 16
    val numFeatures = 20
    val numClasses = 10

    val fc = Linear[Double, CPU](numFeatures, numClasses)
    val input = new Variable(Tensor.randn[Double, CPU](Shape(numSamples, numFeatures)))
    val out = fc(input)
    out.data.shape.asArray shouldBe Array(numSamples, numClasses)
    val dout = new Variable(Tensor.randn[Double, CPU](Shape(numSamples, numClasses)))

    out.backward(dout)
    input.grad.data.shape.asArray shouldBe Array(numSamples, numFeatures)
    println(input.grad)

  }

  it should "evaluate the softmax loss" in {
    val numSamples = 16
    val numFeatures = 20
    val numClasses = 10

    val fc = Linear[Double, CPU](numFeatures, numClasses)
    val input = Variable(Tensor.randn[Double, CPU](Shape(numSamples, numFeatures)))
    val out = fc(input)

    val target = Variable(Tensor.randint[CPU](0, numClasses, Shape(numSamples, 1)))

    val loss = SoftmaxLoss(out, target).forward()

    loss.backward()

    fc.parameters.foreach { p =>
      println(p.data.shape.asArray.toList)
      println(p.grad.data.shape.asArray.toList)
      p.data.shape shouldBe p.grad.data.shape
    }

    input.grad.data.shape.asArray shouldBe Array(numSamples, numFeatures)

  }

  it should "compute a 2 layer fc network with sgd optimizer" in {

    val numSamples = 16
    val numClasses = 5
    val nf1 = 40
    val nf2 = 20

    val target = Variable(Tensor.randint[CPU](0, numClasses, Shape(numSamples, 1)))

    case class Net() extends Module[Double, CPU, Double, Double] {
      val fc1: Linear[Double, CPU] = Linear[Double, CPU](nf1, nf2)
      val fc2: Linear[Double, CPU] = Linear[Double, CPU](nf2, numClasses)
      override def forward(x: Variable[Double, CPU]) = fc2(fc1(x).relu())
    }

    val net = Net()

    val optimizer = SGD[CPU](net.parameters, 1)
    val input = Variable(Tensor.randn[Double, CPU](Shape(numSamples, nf1)))

    for (j <- 0 to 500) {

      optimizer.zeroGrad()

      val output = net(input)
      val loss = SoftmaxLoss(output, target).forward()

      if (j % 100 == 0) {
        val guessed = output.data.argmax(1, true)
        val accuracy = (target.data eq guessed).sum().cast[Double] / numSamples
        println(s"$j: loss: ${loss.data.item()} accuracy: ${accuracy.item()}")
      }

      loss.backward()
      optimizer.step()
    }
    val output = net(input)
    val loss = SoftmaxLoss(output, target).forward()
    val guessed = output.data.argmax(1)
    val accuracy = (target.data eq guessed).sum().cast[Double] / numSamples

    accuracy.item() should be > 0.8
    loss.data.item() should be < 1e-3
  }


  it should "compute a 2 layer fc network with adam optimizer" in {

    val numSamples = 16
    val numClasses = 5
    val nf1 = 40
    val nf2 = 20

    val target = Variable(Tensor.randint[CPU](0, numClasses, Shape(numSamples, 1)))

    case class Net() extends Module[Double, CPU, Double, Double] {
      val fc1: Linear[Double, CPU] = Linear[Double, CPU](nf1, nf2)
      val fc2: Linear[Double, CPU] = Linear[Double, CPU](nf2, numClasses)
      override def forward(x: Variable[Double, CPU]) = fc2(fc1(x).relu())
    }

    val net = Net()

    val optimizer = Adam[CPU](net.parameters, 0.01)
    val input = Variable(Tensor.randn[Double, CPU](Shape(numSamples, nf1)))

    for (j <- 0 to 500) {

      optimizer.zeroGrad()

      val output = net(input)
      val loss = SoftmaxLoss(output, target).forward()

      if (j % 100 == 0) {
        val guessed = output.data.argmax(1, keepdim = true)
//        println(Tensor.summarize(guessed, 20))
//        println(Tensor.summarize((target.data eq guessed).cast[Double], 20))
        val accuracy = (target.data eq guessed).sum().cast[Double] / numSamples
        println(s"$j: loss: ${loss.data.item()} accuracy: ${accuracy.item()}")
      }

      loss.backward()
      optimizer.step()
    }
    val output = net(input)
    val loss = SoftmaxLoss(output, target).forward()
    val guessed = output.data.argmax(1)
    val accuracy = (target.data eq guessed).sum().cast[Double] / numSamples

    accuracy.item() should be > 0.8
    loss.data.item() should be < 1e-3
  }


  it should "correctly process the example from the README" in {
    val numSamples = 128
    val numClasses = 10
    val nf1 = 40
    val nf2 = 20

    // Define the neural network
    case class Net() extends Module[Double, CPU, Double, Double] {

      val fc1: Linear[Double, CPU] = Linear[Double, CPU](nf1, nf2) // an affine operation: y = Wx + b
      val fc2: Linear[Double, CPU] = Linear[Double, CPU](nf2, numClasses) // another one

      // glue the layers with a relu non-linearity: fc1 -> relu -> fc2
      override def forward(x: Variable[Double, CPU]): Variable[Double, CPU] =
        (x ~> fc1).relu() ~> fc2

    }

    // instantiate
    val net = Net()

    // create an optimizer for updating the parameters
    val optimizer = SGD(net.parameters, lr = 0.05)

    // random target and input to train on
    val target = Variable(Tensor.randint[CPU](0, numClasses, Shape(numSamples)))
    val input = Variable(Tensor.randn[Double, CPU](Shape(numSamples, nf1)))

    for (j <- 0 to 1000) {

      // reset the gradients of the parameters
      optimizer.zeroGrad()

      // forward input through the network
      val output = net(input)
      output.shape.asArray shouldBe Array(numSamples, numClasses)

      // calculate the loss
      val loss = SoftmaxLoss(output, target).forward()

      // print loss and accuracy
      if (j % 100 == 0) {
        val guessed = output.data.argmax(1)
        guessed.shape.asArray shouldBe Array(numSamples)
        val accuracy = (target.data eq guessed).sum().cast[Double] / numSamples
        println(s"$j: loss: ${loss.data.item()} accuracy: ${accuracy.item()}")
      }

      // back propagate the derivatives
      loss.backward()

      // update the parameters with the gradients
      optimizer.step()
    }
  }


}