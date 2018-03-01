package scorch.sandbox.rnn

import botkop.numsca._
import botkop.{numsca => ns}
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.{FlatSpec, Matchers}
import scorch.TestUtil._
import scorch.autograd._

import scala.util.Random

class TemporalSoftmaxSpec extends FlatSpec with Matchers {

  Nd4j.setDataType(DataBuffer.Type.DOUBLE)
  // todo: fix random seed in numsca
  ns.rand.setSeed(231)
  Random.setSeed(231)

  def checkLoss(n: Int, t: Int, v: Int, p: Double): Double = {
    val x = 0.001 * ns.randn(n, t, v)
    val y = ns.randint(v, Array(n, t))
    val mask = ns.rand(n, t) <= p
    TemporalSoftmaxFunction(Variable(x), Variable(y), Variable(mask))
      .forward()
      .data
      .squeeze()
  }

  "TemporalSoftmax" should "forward" in {
    checkLoss(100, 1, 10, 1.0) shouldBe 2.3 +- 1e-2
    checkLoss(100, 10, 10, 1.0) shouldBe 23.0 +- 1e-1
    checkLoss(5000, 10, 10, 0.1) shouldBe 2.3 +- 1e-1
  }

  it should "backward pass" in {
    val (n, t, v) = (7, 8, 9)

    val x = Variable(ns.randn(n, t, v))
    val y = Variable(ns.randint(v, Array(n, t)))
    val mask = Variable(ns.rand(n, t) > 0.5)

    val out = TemporalSoftmaxFunction(x, y, mask).forward()
    out.backward()
    val dx = x.grad.data.copy
    // todo: why was this set to None?
    // x.g = None

    def fx(a: Tensor) =
      TemporalSoftmaxFunction(Variable(a), y, mask).forward().data.squeeze()

    val dxNum = evalNumericalGradient(fx, x.data)

    val error = relError(dx, dxNum)
    println(error)
    assert(error < 1e-6)
  }
}


