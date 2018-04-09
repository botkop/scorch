package scorch.nn.cnn

import botkop.{numsca => ns}
import botkop.numsca._ // for :> operator
import org.nd4j.linalg.factory.Nd4j.PadMode
import scorch.nn.Module
import scorch.autograd.{Function, Variable}

case class Conv(w: Variable, b: Variable, stride: Int, pad: Int)
    extends Module(Seq(w, b)) {

  import Conv._

  override def forward(x: Variable): Variable =
    NaiveConvFunction(x, w, b, stride, pad).forward()

}

object Conv {

  def apply(numChannels: Int,
            numFilters: Int,
            filterSize: Int,
            weightScale: Double,
            stride: Int,
            pad: Int): Conv = {
    val w = Variable(
      weightScale * ns.randn(numFilters, numChannels, filterSize, filterSize))
    val b = Variable(ns.zeros(numFilters))
    Conv(w, b, stride, pad)
  }

  case class NaiveConvFunction(x: Variable,
                               w: Variable,
                               b: Variable,
                               stride: Int,
                               pad: Int)
      extends Function {

    val List(numFilters, numChannels, hh, ww) = w.shape
    val List(numDataPoints, _, height, width) = x.shape
    val hPrime: Int = 1 + (height + 2 * pad - hh) / stride
    val wPrime: Int = 1 + (width + 2 * pad - ww) / stride

    val padArea = Array(Array(0, 0), Array(pad, pad), Array(pad, pad))

    override def forward(): Variable = {

      val out = ns.zeros(numDataPoints, numFilters, hPrime, wPrime)

      for (n <- 0 until numDataPoints) {

        val xPad = ns.pad(x.data(n), padArea, PadMode.CONSTANT)

        for (f <- 0 until numFilters) {
          for (hp <- 0 until hPrime) {
            for (wp <- 0 until wPrime) {
              val h1 = hp * stride
              val h2 = h1 + hh
              val w1 = wp * stride
              val w2 = w1 + ww
              val window = xPad(:>, h1 :> h2, w1 :> w2)

              out(n, f, hp, wp) := ns.sum(window * w.data(f)) + b.data(f)
            }
          }
        }
      }
      Variable(out, Some(this))
    }

    override def backward(gradOutput: Variable): Unit = {

      val dOut = gradOutput.data

      val dx = x.grad.data
      val dw = w.grad.data
      val db = b.grad.data

      for (n <- 0 until numDataPoints) {
        val dxPad = ns.pad(dx(n), padArea, PadMode.CONSTANT)
        val xPad = ns.pad(x.data(n), padArea, PadMode.CONSTANT)
        for (f <- 0 until numFilters) {
          for (hp <- 0 until hPrime) {
            for (wp <- 0 until wPrime) {
              val h1 = hp * stride
              val h2 = h1 + hh
              val w1 = wp * stride
              val w2 = w1 + ww
              val d = dOut(n, f, hp, wp)
              dxPad(:>, h1 :> h2, w1 :> w2) += w.data(f) * d
              dw(f) += xPad(:>, h1 :> h2, w1 :> w2) * d
              db(f) += d
            }
          }
        }
        dx(n) := dxPad(:>, 1 :> -1, 1 :> -1)
      }
    }
  }
}
