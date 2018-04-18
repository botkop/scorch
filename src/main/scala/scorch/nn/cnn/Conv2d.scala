package scorch.nn.cnn

import botkop.{numsca => ns}
import botkop.numsca._ // for :> operator
import org.nd4j.linalg.factory.Nd4j.PadMode
import scorch.nn.Module
import scorch.autograd.{Function, Variable}

case class Conv2d(w: Variable, b: Variable, pad: Int, stride: Int)
    extends Module(Seq(w, b)) {

  import Conv2d._

  def outputShape(inputShape: List[Int]): List[Int] =
    Conv2d.outputShape(inputShape, w.shape, pad, stride)

  override def forward(x: Variable): Variable =
    NaiveConv2dFunction(x, w, b, pad, stride).forward()
}

object Conv2d {

  def apply(numChannels: Int,
            numFilters: Int,
            filterSize: Int,
            weightScale: Double,
            pad: Int,
            stride: Int): Conv2d = {
    val w = Variable(
      weightScale * ns.randn(numFilters, numChannels, filterSize, filterSize))
    val b = Variable(ns.zeros(numFilters))
    Conv2d(w, b, pad, stride)
  }

  def outputShape(xShape: List[Int],
                  wShape: List[Int],
                  pad: Int,
                  stride: Int): List[Int] = {
    val List(numFilters, _, hh, ww) = wShape
    val List(numSamples, _, height, width) = xShape
    val hPrime: Int = 1 + (height + 2 * pad - hh) / stride
    val wPrime: Int = 1 + (width + 2 * pad - ww) / stride
    List(numSamples, numFilters, hPrime, wPrime)
  }

  case class NaiveConv2dFunction(x: Variable,
                                 w: Variable,
                                 b: Variable,
                                 pad: Int,
                                 stride: Int)
      extends Function {

    val List(numDataPoints, numFilters, hPrime, wPrime) =
      outputShape(x.shape, w.shape, pad, stride)
    val List(hh, ww) = w.shape.takeRight(2)

    val padArea = Array(Array(0, 0), Array(pad, pad), Array(pad, pad))

    override def forward(): Variable = {

      val out = ns.zeros(numDataPoints, numFilters, hPrime, wPrime)

      for {
        n <- 0 until numDataPoints
        xPad = ns.pad(x.data(n), padArea, PadMode.CONSTANT)

        f <- 0 until numFilters
        wf = w.data(f)
        bf = b.data(f)

        hp <- 0 until hPrime
        h1 = hp * stride
        h2 = h1 + hh

        wp <- 0 until wPrime
        w1 = wp * stride
        w2 = w1 + ww
      } {
        val window = xPad(:>, h1 :> h2, w1 :> w2)
        out(n, f, hp, wp) := ns.sum(window * wf) + bf
      }

      Variable(out, Some(this))
    }

    override def backward(gradOutput: Variable): Unit = {

      val dOut = gradOutput.data

      val dx = zerosLike(x.data)
      val dw = zerosLike(w.data)
      val db = zerosLike(b.data)

      for {
        n <- 0 until numDataPoints
        dxPad = ns.pad(dx(n), padArea, PadMode.CONSTANT)
        xPad = ns.pad(x.data(n), padArea, PadMode.CONSTANT)

        f <- 0 until numFilters
        wf = w.data(f)

        hp <- 0 until hPrime
        h1 = hp * stride
        h2 = h1 + hh

        wp <- 0 until wPrime
        w1 = wp * stride
        w2 = w1 + ww
      } {
        val d = dOut(n, f, hp, wp)
        dxPad(:>, h1 :> h2, w1 :> w2) += wf * d
        dw(f) += xPad(:>, h1 :> h2, w1 :> w2) * d
        db(f) += d
        dx(n) := dxPad(:>, 1 :> -1, 1 :> -1)
      }

      x.backward(Variable(dx))
      w.backward(Variable(dw))
      b.backward(Variable(db))
    }
  }
}
