package scorch.nn

import botkop.numsca.Tensor
import botkop.{numsca => ns}
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.{FlatSpec, Matchers}
import scorch.TestUtil._
import scorch.autograd._

import scala.language.postfixOps

class RnnSpec extends FlatSpec with Matchers {

  Nd4j.setDataType(DataBuffer.Type.DOUBLE)
  ns.rand.setSeed(231)

  "Rnn" should "step forward" in {

    val (n, d, h) = (3, 10, 4)

    val x = Variable(ns.linspace(-0.4, 0.7, num = n * d).reshape(n, d))
    val prevH = Variable(ns.linspace(-0.2, 0.5, num = n * h).reshape(n, h))
    val wX = Variable(ns.linspace(-0.1, 0.9, num = d * h).reshape(d, h))
    val wH = Variable(ns.linspace(-0.3, 0.7, num = h * h).reshape(h, h))
    val b = Variable(ns.linspace(-0.2, 0.4, num = h).reshape(1, h))

    val nextH = RnnFunction.stepForward(x, prevH, wX, wH, b)

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
    val (n, d, hSize) = (4, 5, 6)

    val x = Variable(ns.randn(n, d))
    val h = Variable(ns.randn(n, hSize))
    val wX = Variable(ns.randn(d, hSize))
    val wH = Variable(ns.randn(hSize, hSize))
    val b = Variable(ns.randn(1, hSize))

    val out = RnnFunction.stepForward(x, h, wX, wH, b)

    // loss simulation
    val dNextH = Variable(ns.randn(out.shape: _*))
    out.backward(dNextH)

    val dx = x.grad.get.data.copy()
    val dh = h.grad.get.data.copy()
    val dwX = wX.grad.get.data.copy()
    val dwH = wH.grad.get.data.copy()
    val db = b.grad.get.data.copy()

    def fx(t: Tensor): Tensor =
      RnnFunction.stepForward(Variable(t), h, wX, wH, b).data

    def fh(t: Tensor): Tensor =
      RnnFunction.stepForward(x, Variable(t), wX, wH, b).data

    def fwX(t: Tensor): Tensor =
      RnnFunction.stepForward(x, h, Variable(t), wH, b).data

    def fwH(t: Tensor): Tensor =
      RnnFunction.stepForward(x, h, wX, Variable(t), b).data

    def fb(t: Tensor): Tensor =
      RnnFunction.stepForward(x, h, wX, wH, Variable(t)).data

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

  it should "forward pass" in {
    val (n, t, d, h) = (2, 3, 4, 5)

    val x = Variable(ns.linspace(-0.1, 0.3, num = n * t * d).reshape(n, t, d))
    val h0 = Variable(ns.linspace(-0.3, 0.1, num = n * h).reshape(n, h))
    val wX = Variable(ns.linspace(-0.2, 0.4, num = d * h).reshape(d, h))
    val wH = Variable(ns.linspace(-0.4, 0.1, num = h * h).reshape(h, h))
    val b = Variable(ns.linspace(-0.7, 0.1, num = h))

    val out = RnnFunction(x, h0, wX, wH, b).forward()

    val expectedH = ns
      .array(-0.42070749, -0.27279261, -0.11074945, 0.05740409, 0.22236251,
        -0.39525808, -0.22554661, -0.0409454, 0.14649412, 0.32397316,
        -0.42305111, -0.24223728, -0.04287027, 0.15997045, 0.35014525,
        -0.55857474, -0.39065825, -0.19198182, 0.02378408, 0.23735671,
        -0.27150199, -0.07088804, 0.13562939, 0.33099728, 0.50158768,
        -0.51014825, -0.30524429, -0.06755202, 0.17806392, 0.40333043)
      .reshape(n, t, h)

    val error = relError(out.data, expectedH)
    println(error)
    assert(error < 1e-7)
  }

  it should "backward pass" in {

    val (n, d, t, h) = (2, 3, 2, 5)

    val x = Variable(ns.randn(n, t, d), name = Some("x"))
    val h0 = Variable(ns.randn(n, h), name = Some("h0"))
    val wX = Variable(ns.randn(d, h), name = Some("wX"))
    val wH = Variable(ns.randn(h, h), name = Some("wH"))
    val b = Variable(ns.randn(h), name = Some("b"))
    val rnn = RnnFunction(x, h0, wX, wH, b)

    val out = rnn.forward()
    val dOut = Variable(ns.randn(out.shape.toArray))
    out.backward(dOut)

    val dx = x.grad.get.data
    def fx(t: Tensor): Tensor = RnnFunction(Variable(t), h0, wX, wH, b).forward().data
    val dxNum = evalNumericalGradientArray(fx, x.data, dOut.data)
    val dxError = relError(dx, dxNum)
    println(dxError)
    assert(dxError < 1e-7)

    val dh0 = h0.grad.get.data
    def fh0(t: Tensor): Tensor = RnnFunction(x, Variable(t), wX, wH, b).forward().data
    val dh0Num = evalNumericalGradientArray(fh0, h0.data, dOut.data)
    val dh0Error = relError(dh0, dh0Num)
    println(dh0Error)
    assert(dh0Error < 1e-7)

    val dwX = wX.grad.get.data
    def fwX(t: Tensor): Tensor = RnnFunction(x, h0, Variable(t), wH, b).forward().data
    val dwXNum = evalNumericalGradientArray(fwX, wX.data, dOut.data)
    val dwXError = relError(dwX, dwXNum)
    println(dwXError)
    assert(dwXError < 1e-7)

    val dwH = wH.grad.get.data
    def fwH(t: Tensor): Tensor = RnnFunction(x, h0, wX, Variable(t), b).forward().data
    val dwHNum = evalNumericalGradientArray(fwH, wH.data, dOut.data)
    val dwHError = relError(dwH, dwHNum)
    println(dwHError)
    assert(dwHError < 1e-7)

    val db = b.grad.get.data
    def fb(t: Tensor): Tensor = RnnFunction(x, h0, wX, wH, Variable(t)).forward().data
    val dbNum = evalNumericalGradientArray(fb, b.data, dOut.data)
    val dbError = relError(db, dbNum)
    println(dbError)
    assert(dbError < 1e-7)



  }

}
