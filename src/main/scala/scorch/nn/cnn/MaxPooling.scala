package scorch.nn.cnn

import botkop.numsca._
import botkop.{numsca => ns}
import scorch.autograd.{Function, Variable}
import scorch.nn.Module
import scorch.nn.cnn.MaxPooling.NaiveMaxPoolingFunction

// does not have learnable parameters, so probably not a module
case class MaxPooling(poolHeight: Int, poolWidth: Int, stride: Int) extends Module {
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

      for {
        n <- 0 until numDataPoints
        c <- 0 until numChannels

        h <- 0 until hPrime
        h1 = h * stride
        h2 = h1 + poolHeight

        w <- 0 until wPrime
        w1 = w * stride
        w2 = w1 + poolWidth
      } {
        val window = x.data(n, :>, h1 :> h2, w1 :> w2)
        out(n, :>, h, w) := ns.max(
          window.reshape(numChannels, poolHeight * poolWidth),
          axis = 1)
      }

      Variable(out, Some(this))
    }

    override def backward(gradOutput: Variable): Unit = {
      val dx = ns.zerosLike(x.data)
      val dOut = gradOutput.data

      for {
        n <- 0 until numDataPoints
        c <- 0 until numChannels

        h <- 0 until hPrime
        h1 = h * stride
        h2 = h1 + poolHeight

        w <- 0 until wPrime
        w1 = w * stride
        w2 = w1 + poolWidth
      } {
        val window = x.data(n, c, h1 :> h2, w1 :> w2)
        // next works if there is exactly 1 maximum
        // todo: perhaps add check for this
        val upd = (window == ns.max(window)) * dOut(n, c, h, w)
        dx(n, c, h1 :> h2, w1 :> w2) := upd
      }

      x.backward(Variable(dx))
    }

  }
}
