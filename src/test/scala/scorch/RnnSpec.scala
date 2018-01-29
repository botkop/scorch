package scorch

import botkop.numsca.Tensor
import botkop.{numsca => ns}
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.{FlatSpec, Matchers}
import scorch.autograd._
import scorch.TestUtil._

class RnnSpec extends FlatSpec with Matchers {

  case class Rnn() {
    def stepForward(x: Variable,
                    prevH: Variable,
                    wX: Variable,
                    wH: Variable,
                    b: Variable): Variable = {
      val xWx = x dot wX
      val prevHwH = prevH dot wH
      val nextH = tanh(xWx + prevHwH + b)
      nextH
    }
  }

  "Rnn" should "step forward" in {
    Nd4j.setDataType(DataBuffer.Type.DOUBLE)

    val (n, d, h) = (3, 10, 4)

    val x = Variable(ns.linspace(-0.4, 0.7, num = n * d).reshape(n, d))
    val prevH = Variable(ns.linspace(-0.2, 0.5, num = n * h).reshape(n, h))
    val wX = Variable(ns.linspace(-0.1, 0.9, num = d * h).reshape(d, h))
    val wH = Variable(ns.linspace(-0.3, 0.7, num = h * h).reshape(h, h))
    val b = Variable(ns.linspace(-0.2, 0.4, num = h).reshape(1, h))

    val nextH = Rnn().stepForward(x, prevH, wX, wH, b)

    val expectedNextH = ns
      .array(
        -0.58172089, -0.50182032, -0.41232771, -0.31410098, //
        0.66854692, 0.79562378, 0.87755553, 0.92795967, //
        0.97934501, 0.99144213, 0.99646691, 0.99854353 //
      )
      .reshape(3, 4)

    val re = relError(expectedNextH, nextH.data)
    println(re)

    assert(re < 1e-8)
  }

  it should "step backward" in {
    Nd4j.setDataType(DataBuffer.Type.DOUBLE)

    val (n, d, hSize) = (4, 5, 6)

    val x = Variable(ns.randn(n, d))
    val h = Variable(ns.randn(n, hSize))
    val wX = Variable(ns.randn(d, hSize))
    val wH = Variable(ns.randn(hSize, hSize))
    val b = Variable(ns.randn(1, hSize))

    val rnn = Rnn()
    val out = rnn.stepForward(x, h, wX, wH, b)

    // loss simulation
    val dNextH = Variable(ns.randn(out.shape: _*))
    out.backward(dNextH)

    val dx = x.grad.get.data.copy()
    val dh = h.grad.get.data.copy()
    val dwX = wX.grad.get.data.copy()
    val dwH = wH.grad.get.data.copy()
    val db = b.grad.get.data.copy()

    def fx(t: Tensor): Tensor =
      rnn.stepForward(Variable(t), h, wX, wH, b).data
    def fh(t: Tensor): Tensor =
      rnn.stepForward(x, Variable(t), wX, wH, b).data
    def fwX(t: Tensor): Tensor =
      rnn.stepForward(x, h, Variable(t), wH, b).data
    def fwH(t: Tensor): Tensor =
      rnn.stepForward(x, h, wX, Variable(t), b).data
    def fb(t: Tensor): Tensor =
      rnn.stepForward(x, h, wX, wH, Variable(t)).data

    val dxNum = evalNumericalGradientArray(fx, x.data, dNextH.data)
    val dhNum = evalNumericalGradientArray(fh, h.data, dNextH.data)
    val dwXNum = evalNumericalGradientArray(fwX, wX.data, dNextH.data)
    val dwHNum = evalNumericalGradientArray(fwH, wH.data, dNextH.data)
    val dbNum = evalNumericalGradientArray(fb, b.data, dNextH.data)

    val dxError = relError(dx, dxNum)
    val dhError = relError(dh, dhNum)
    val dwXError = relError(dwX, dwXNum)
    val dwHError = relError(dwH, dwHNum)
    val dbError = relError(db, dbNum)

    println(s"dxError = $dxError")
    println(s"dhError = $dhError")
    println(s"dwXError = $dwXError")
    println(s"dwHError = $dwHError")
    println(s"dbError = $dbError")

    assert(dxError < 1e-8)
    assert(dhError < 1e-8)
    assert(dwXError < 1e-8)
    assert(dwHError < 1e-8)
    assert(dbError < 1e-8)

  }

}
