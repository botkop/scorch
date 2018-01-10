package botkop

import botkop.autograd.Variable
import botkop.nn.Linear
import botkop.{numsca => ns}
import org.scalatest.{FlatSpec, Matchers}

class ModuleSpec extends FlatSpec with Matchers {

  "A Module" should "compute a simple linear network" in {

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

  it should "evaluate the loss" in {
    val numSamples = 16
    val numFeatures = 20
    val numClasses = 10

    val fc = Linear(numFeatures, numClasses)
    val input = Variable(ns.randn(numSamples, numFeatures))
    val out = fc(input)

    val target = Variable(ns.randint(numClasses, Array(numSamples, 1)))

    val dout = nn.softmax(out, target)

    println(dout.data)

    dout.backward()

    input.grad.get.data.shape shouldBe Array(numSamples, numFeatures)
    println(input.grad.get)
    println(fc.weights.grad)
  }
}
