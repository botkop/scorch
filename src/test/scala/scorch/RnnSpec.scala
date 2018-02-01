package scorch

import botkop.numsca.{:>, Tensor}
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

  case class Rnn() {
    def stepForward(x: Variable,
                    prevH: Variable,
                    wX: Variable,
                    wH: Variable,
                    b: Variable): Variable = {

      // x:     shape(N, D)
      // prevH: shape(N, H)
      // wX:    shape(D, H)
      // wH:    shape(H, H)
      // b:     shape(H, 1)

      val xWx = x dot wX
      val prevHwH = prevH dot wH
      val nextH = tanh(xWx + prevHwH + b)
      nextH
    }

    def forward(xs: List[Variable],
                h0: Variable,
                wX: Variable,
                wH: Variable,
                b: Variable): List[Variable] = {
      // nSize: minibatch of N sequences
      // tSize: input sequence composed of T vectors
      // dSize: dimension of the vector
      // hSize: hidden size

      xs.foldLeft(List.empty[Variable]) {
        case (hs: List[Variable], x: Variable) =>
          hs :+ stepForward(x, hs.lastOption.getOrElse(h0), wX, wH, b)
      }
    }

    def backward(gs: List[Variable]): Unit = {

    }

  }

  "Rnn" should "step forward" in {

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

  it should "forward pass" in {

    val (n, t, d, h) = (2, 3, 4, 5)

    val xVector = ns.linspace(-0.1, 0.3, num = n * t * d).reshape(n, t, d)

    val xs: List[Variable] = (0 until t) map { i =>
      Variable(xVector(:>, i, :>).reshape(n, d))
    } toList

    val h0 = Variable(ns.linspace(-0.3, 0.1, num = n * h).reshape(n, h))
    val wX = Variable(ns.linspace(-0.2, 0.4, num = d * h).reshape(d, h))
    val wH = Variable(ns.linspace(-0.4, 0.1, num = h * h).reshape(h, h))
    val b = Variable(ns.linspace(-0.7, 0.1, num = h))

    val rnn = Rnn()

    val hidden = rnn.forward(xs, h0, wX, wH, b)

    hidden.foreach { h =>
      println(h.data)
      println
    }

    val expectedH = ns.array(
      -0.42070749, -0.27279261, -0.11074945, 0.05740409, 0.22236251, //
      -0.39525808, -0.22554661, -0.0409454, 0.14649412, 0.32397316, //
      -0.42305111, -0.24223728, -0.04287027, 0.15997045, 0.35014525, //
      //
      -0.55857474, -0.39065825, -0.19198182, 0.02378408, 0.23735671, //
      -0.27150199, -0.07088804, 0.13562939, 0.33099728, 0.50158768, //
      -0.51014825, -0.30524429, -0.06755202, 0.17806392, 0.40333043 //
    ).reshape(n, t, h)

    for (i <- 0 until t) {
      val error = relError(hidden(i).data, expectedH(:>, i, :>).reshape(n, h))
      println(error)
      assert(error < 1e-7)
    }

  }

  it should "backward pass" in {

    val (n, d, t, h) = (2, 3, 2, 5)

    val xs: List[Variable] = List.fill(t) {
      Variable(ns.randn(n, d))
    }

    val h0 = Variable(ns.randn(n, h))
    val wX = Variable(ns.randn(d, h))
    val wH = Variable(ns.randn(h, h))
    val b = Variable(ns.randn(h))
    val rnn = Rnn()

    val out = rnn.forward(xs, h0, wX, wH, b)

    val dOut = List.fill(out.length) {
      Variable(ns.randn(out.head.shape.toArray))
    }

    out.zip(dOut).reverse.foreach { case (o, g) => o.backward(g) }

    // def fh0(t: Tensor): Tensor = rnn.forward(xs, Variable(t), wX, wH, b).head.data
    // val dFh0 = evalNumericalGradientArray(fh0, h0.data, h0.grad.get.data)
    // println(dFh0)

    println(h0.grad.get)

  }

}
