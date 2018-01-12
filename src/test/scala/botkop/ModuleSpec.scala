package botkop

import botkop.autograd.Variable
import botkop.nn.{Linear, Module}
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

    val loss = nn.softmax(out, target)

    println(loss.data)

    loss.backward()

    fc.parameters().foreach { p =>
      println(p.data.shape.toList)
      println(p.grad.get.data.shape.toList)
      p.data.shape shouldBe p.grad.get.data.shape
    }

    input.grad.get.data.shape shouldBe Array(numSamples, numFeatures)
    //println(input.grad.get)
    //println(fc.weights.grad)

  }

  it should "say miauou" in {

    case class Net() extends Module {
      val fc1 = Linear(40, 20)
      val fc2 = Linear(20, 10)
      override def subModules(): Seq[Module] = Seq(fc1, fc2)
      override def forward(x: Variable): Variable = fc2(nn.relu(fc1(x)))
    }

    val n = Net()
    val input = Variable(ns.randn(16, 40))
    val output = n(input)
    val target = Variable(ns.randint(10, Array(16, 1)))
    val loss = nn.softmax(output, target)
    println(loss)
    loss.backward()

    val lr = 0.01
    n.parameters().foreach { p =>
      println(p.data.shape.toList)
      println(p.grad.get.data.shape.toList)
      p.data.shape shouldBe p.grad.get.data.shape
      p.data -= p.grad.get.data * lr
    }

  }
}
