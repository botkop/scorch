package scorch.nn

import botkop.{numsca => ns}
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.{FlatSpec, Matchers}
import scorch.autograd.Variable
import scorch._
import scorch.autograd.Variable._
import scorch.TestUtil._
import scorch.nn.Parallelize.ParallelizeFunction

import scala.concurrent.duration._
import scala.language.postfixOps
import scala.util.Random

class ParallelizeSpec extends FlatSpec with Matchers {

  Nd4j.setDataType(DataBuffer.Type.DOUBLE)
  ns.rand.setSeed(231)
  Random.setSeed(231)

  "A Parallelization module" should "correctly reconstitute output in forward pass" in {
    val batchSize = 32
    val numFeatures = 20

    case class Net() extends Module {
      override def forward(x: Variable): Variable = Variable(x.data.copy())
    }

    val net = Net()
    val p = Parallelize(net, parallelism = 4, timeOut = 2 seconds)
    val input = Variable(ns.randn(batchSize, numFeatures))
    val yHat = p(input)

    assert(yHat.data.sameElements(input.data))
    assert(yHat.data.sameShape(input.data))
  }

  it should "update the gradients of the module" in {
    val batchSize = 32
    val numFeatures = 20
    val numClasses = 10

    case class Net() extends Module {
      val fc = Linear(numFeatures, numClasses)
      override def forward(x: Variable): Variable = x ~> fc ~> relu
    }
    val net = Net()

    val parCopies = net.parameters.map { p =>
      Variable(p.data.copy())
    }

    val p = Parallelize(net, parallelism = 4, timeOut = 2 seconds)

    val input = Variable(ns.randn(batchSize, numFeatures))

    val yHat = p(input)

    val target = Variable(ns.randint(numClasses, Array(batchSize, 1)))
    val loss = softmaxLoss(yHat, target)

    loss.backward()

    net.parameters.map(_.grad).foreach(println)

    net.parameters.foreach { p =>
      assert(p.grad.data.sameShape(p.data))
      assert(ns.sum(p.grad.data) != 0.0)
    }

  }

  "The parallelization function" should "calulate the correct gradients" in {
    val batchSize = 100
    val numFeatures = 20
    val numClasses = 10
    case class Net() extends Module {
      val fc = Linear(numFeatures, numClasses)
      override def forward(x: Variable): Variable = x ~> fc ~> relu
    }
    val net = Net()

    def f(a: Variable) =
      ParallelizeFunction(a,
                          module = net,
                          parallelism = 8,
                          timeOut = 20 seconds)
        .forward()

    val input = Variable(ns.randn(batchSize, numFeatures))
    oneOpGradientCheck(f, input, 1e-7)
  }

}
