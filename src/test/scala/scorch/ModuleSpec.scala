package scorch

import scorch.autograd.Variable
import scorch.nn._
import botkop.{numsca => ns}
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

class ModuleSpec extends FlatSpec with Matchers {

  "A Module" should "compute a single pass through a linear network" in {

    val numSamples = 16
    val numFeatures = 20
    val numClasses = 10

    val fc = Linear(numFeatures, numClasses)
    val input = Variable(ns.randn(numSamples, numFeatures))
    val out = fc(input)
    out.data.shape shouldBe Array(numSamples, numClasses)
    val dout = Variable(ns.randn(numSamples, numClasses))

    out.backward(dout)
    input.grad.get.data.shape shouldBe Array(numSamples, numFeatures)
    println(input.grad)

  }

  it should "evaluate the softmax loss" in {
    val numSamples = 16
    val numFeatures = 20
    val numClasses = 10

    val fc = Linear(numFeatures, numClasses)
    val input = Variable(ns.randn(numSamples, numFeatures))
    val out = fc(input)

    val target = Variable(ns.randint(numClasses, Array(numSamples, 1)))

    val loss = nn.softmax(out, target)

    loss.backward()

    fc.parameters().foreach { p =>
      println(p.data.shape.toList)
      println(p.grad.get.data.shape.toList)
      p.data.shape shouldBe p.grad.get.data.shape
    }

    input.grad.get.data.shape shouldBe Array(numSamples, numFeatures)

  }

  it should "compute a 2 layer fc network with sgd optimizer" in {
    ns.rand.setSeed(231)
    Random.setSeed(231)

    val numSamples = 16
    val numClasses = 5
    val nf1 = 40
    val nf2 = 20

    val target = Variable(ns.randint(numClasses, Array(numSamples, 1)))

    case class Net() extends Module {
      val fc1 = Linear(nf1, nf2)
      val fc2 = Linear(nf2, numClasses)
      override def subModules(): Seq[Module] = Seq(fc1, fc2)
      override def forward(x: Variable): Variable = fc2(nn.relu(fc1(x)))
    }

    val n = Net()

    val optimizer = SGD(n.parameters(), lr = 1)

    val input = Variable(ns.randn(numSamples, nf1))

    for (j <- 0 to 500) {

      optimizer.zeroGrad()

      val output = n(input)
      val loss = nn.softmax(output, target)

      if (j % 100 == 0) {
        val guessed = ns.argmax(output.data, axis = 1)
        val accuracy = ns.sum(target.data == guessed) / numSamples
        println(s"$j: loss: ${loss.data.squeeze()} accuracy: $accuracy")
      }

      loss.backward()
      optimizer.step()
    }

    val output = n(input)
    val loss = nn.softmax(output, target)
    val guessed = ns.argmax(output.data, axis = 1)
    val accuracy = ns.sum(target.data == guessed) / numSamples

    accuracy should be > 0.8
    loss.data.squeeze should be < 1e-3
  }

  it should "compute a 2 layer fc network with adam optimizer" in {
    ns.rand.setSeed(231)
    Random.setSeed(231)

    val numSamples = 16
    val numClasses = 10
    val nf1 = 40
    val nf2 = 20

    val target = Variable(ns.randint(numClasses, Array(numSamples, 1)))

    case class Net() extends Module {
      val fc1 = Linear(nf1, nf2)
      val fc2 = Linear(nf2, numClasses)
      override def subModules(): Seq[Module] = Seq(fc1, fc2)
      override def forward(x: Variable): Variable = fc2(nn.relu(fc1(x)))
    }

    val n = Net()

    val optimizer = Adam(n.parameters(), lr = 0.01)

    val input = Variable(ns.randn(numSamples, nf1))

    for (j <- 0 to 500) {

      optimizer.zeroGrad()

      val output = n(input)
      val loss = nn.softmax(output, target)

      if (j % 100 == 0) {
        val guessed = ns.argmax(output.data, axis = 1)
        val accuracy = ns.sum(target.data == guessed) / numSamples
        println(s"$j: loss: ${loss.data.squeeze()} accuracy: $accuracy")
      }

      loss.backward()
      optimizer.step()
    }

    val output = n(input)
    val loss = nn.softmax(output, target)
    val guessed = ns.argmax(output.data, axis = 1)
    val accuracy = ns.sum(target.data == guessed) / numSamples

    accuracy should be > 0.6
    loss.data.squeeze should be < 1e-3
  }

  it should "compute a 2 layer fc network with nesterov optimizer" in {
    ns.rand.setSeed(231)
    Random.setSeed(231)

    val numSamples = 128
    val numClasses = 10
    val nf1 = 40
    val nf2 = 20

    val target = Variable(ns.randint(numClasses, Array(numSamples, 1)))

    case class Net() extends Module {
      val fc1 = Linear(nf1, nf2)
      val fc2 = Linear(nf2, numClasses)
      override def subModules(): Seq[Module] = Seq(fc1, fc2)
      override def forward(x: Variable): Variable = fc2(nn.relu(fc1(x)))
    }

    val n = Net()

    val optimizer = Nesterov(n.parameters(), lr = 0.01)

    val input = Variable(ns.randn(numSamples, nf1))

    for (j <- 0 to 500) {

      optimizer.zeroGrad()

      val output = n(input)
      val loss = nn.softmax(output, target)

      if (j % 100 == 0) {
        val guessed = ns.argmax(output.data, axis = 1)
        val accuracy = ns.sum(target.data == guessed) / numSamples
        println(s"$j: loss: ${loss.data.squeeze()} accuracy: $accuracy")
      }

      loss.backward()
      optimizer.step()
    }

    val output = n(input)
    val loss = nn.softmax(output, target)
    val guessed = ns.argmax(output.data, axis = 1)
    val accuracy = ns.sum(target.data == guessed) / numSamples

    accuracy should be > 0.7
    loss.data.squeeze should be < 0.3
  }

  it should "compute a 2 layer fc network with sgd optimizer and dropout" in {
    ns.rand.setSeed(231)
    Random.setSeed(231)

    val numSamples = 16
    val numClasses = 5
    val nf1 = 40
    val nf2 = 20

    val target = Variable(ns.randint(numClasses, Array(numSamples, 1)))

    case class Net() extends Module {
      val fc1 = Linear(nf1, nf2)
      val fc2 = Linear(nf2, numClasses)
      override def subModules(): Seq[Module] = Seq(fc1, fc2)
      override def forward(x: Variable): Variable = fc2(dropout(nn.relu(fc1(x)), p = 0.2, train = true))
    }

    val n = Net()

    val optimizer = SGD(n.parameters(), lr = 1)

    val input = Variable(ns.randn(numSamples, nf1))

    for (j <- 0 to 500) {

      optimizer.zeroGrad()

      val output = n(input)
      val loss = nn.softmax(output, target)

      if (j % 100 == 0) {
        val guessed = ns.argmax(output.data, axis = 1)
        val accuracy = ns.sum(target.data == guessed) / numSamples
        println(s"$j: loss: ${loss.data.squeeze()} accuracy: $accuracy")
      }

      loss.backward()
      optimizer.step()
    }

    val output = n(input)
    val loss = nn.softmax(output, target)
    val guessed = ns.argmax(output.data, axis = 1)
    val accuracy = ns.sum(target.data == guessed) / numSamples

    accuracy should be > 0.8
    loss.data.squeeze should be < 1e-3

  }

}
