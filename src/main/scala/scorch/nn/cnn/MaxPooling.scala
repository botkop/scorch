package scorch.nn.cnn

import scorch.autograd.Variable
import scorch.nn.Module
import botkop.{numsca => ns}
import botkop.numsca._
import org.nd4j.linalg.factory.Nd4j.PadMode
import scorch.nn.Module
import scorch.autograd.{Function, Variable}
import scorch.nn.cnn.MaxPooling.NaiveMaxPoolingFunction

class MaxPooling(poolHeight: Int, poolWidth: Int, stride: Int) extends Module {
  override def forward(x: Variable): Variable =
    NaiveMaxPoolingFunction(x, poolHeight, poolWidth, stride).forward()
}

object MaxPooling {

  case class NaiveMaxPoolingFunction(x: Variable,
                                     poolHeight: Int,
                                     poolWidth: Int,
                                     stride: Int)
      extends Function {

    val List(numDataPoints, numChannels, height, width) = x.shape
    val hPrime: Int = 1 + (height - poolHeight) / stride
    val wPrime: Int = 1 + (width - poolWidth) / stride

    override def forward(): Variable = {

      val out = ns.zeros(numDataPoints, numChannels, hPrime, wPrime)

      for (n <- 0 until numDataPoints) {
        for (h <- 0 until hPrime) {
          for (w <- 0 until wPrime) {
            val h1 = h * stride
            val h2 = h1 + poolHeight
            val w1 = w * stride
            val w2 = w1 + poolWidth
            val window = x.data(n, :>, h1 :> h2, w1 :> w2)
            out(n, :>, h, w) := ns.max(
              window.reshape(numChannels, poolHeight * poolWidth),
              axis = 1)
          }
        }
      }
      Variable(out, Some(this))
    }

    override def backward(gradOutput: Variable): Unit = {
      // val dx = x.grad.data
      val dx = ns.zerosLike(x.data)
      val dOut = gradOutput.data

      for (n <- 0 until numDataPoints) {
        for (c <- 0 until numChannels) {
          for (h <- 0 until hPrime) {
            for (w <- 0 until wPrime) {
              val h1 = h * stride
              val h2 = h1 + poolHeight
              val w1 = w * stride
              val w2 = w1 + poolWidth
              val window = x.data(n, c, h1 :> h2, w1 :> w2).reshape(1, 1, poolHeight, poolWidth)

              // val flat = window.reshape(poolHeight * poolWidth, 1)
              val flat = window.reshape(window.shape.product, 1)
              val maxIndex = ns.argmax(flat).squeeze().toInt
              val mask = ns.zerosLike(flat)
              mask(maxIndex, 0) := 1
              val upd = mask.reshape(1, 1, poolHeight, poolWidth) * dOut(n, c, h, w)

              // println(upd)
              // println

              dx(n, c, h1 :> h2, w1 :> w2) := upd
            }
          }
        }
      }
      x.grad.data := dx
    }

  }
}
