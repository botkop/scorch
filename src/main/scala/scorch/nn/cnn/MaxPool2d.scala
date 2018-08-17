package scorch.nn.cnn

import botkop.numsca._
import botkop.{numsca => ns}
import scorch.autograd.{Function, Variable}
import scorch.nn.Module
import scorch.nn.cnn.MaxPool2d._

// does not have learnable parameters, so not really a module
case class MaxPool2d(poolHeight: Int, poolWidth: Int, stride: Int)
    extends Module {

  def outputShape(inputShape: List[Int]): List[Int] =
    MaxPool2d.outputShape(inputShape, poolHeight, poolWidth, stride)

  override def forward(x: Variable): Variable = {
    val List(height, width) = x.shape.takeRight(2)
    if ((poolHeight == poolWidth) &&
        (poolHeight == stride) &&
        (height % poolHeight == 0) &&
        (width % poolWidth == 0))
      ReshapeMaxPool2dFunction(x, poolHeight, poolWidth, stride).forward()
    else
      NaiveMaxPool2dFunction(x, poolHeight, poolWidth, stride).forward()
  }
}

object MaxPool2d {

  def apply(poolSize: Int, stride: Int): MaxPool2d =
    MaxPool2d(poolSize, poolSize, stride)

  def outputShape(inputShape: List[Int],
                  poolHeight: Int,
                  poolWidth: Int,
                  stride: Int): List[Int] = {
    val List(n, c, h, w) = inputShape
    val hPrime = 1 + (h - poolHeight) / stride
    val wPrime = 1 + (w - poolWidth) / stride
    List(n, c, hPrime, wPrime)
  }

  case class ReshapeMaxPool2dFunction(x: Variable,
                                      poolHeight: Int,
                                      poolWidth: Int,
                                      stride: Int)
      extends Function {

    val List(height, width) = x.shape.takeRight(2)

    assert(poolHeight == poolWidth && poolHeight == stride,
           "Invalid pool params")
    assert(height % poolHeight == 0)
    assert(width % poolWidth == 0)

    val List(batchSize, channels, hPrime, wPrime) =
      outputShape(x.shape, poolHeight, poolWidth, stride)

    val xReshaped: Tensor = x.data.reshape(batchSize,
                                           channels,
                                           height / poolHeight,
                                           poolHeight,
                                           width / poolWidth,
                                           poolWidth)

    // todo implement amax in numsca
    val out = new Tensor(xReshaped.array.max(3).max(4))

    override def forward(): Variable = Variable(out, Some(this))

    override def backward(gradOutput: Variable): Unit = {

      val bwStart = System.currentTimeMillis()

      val dx = ns.zerosLike(x.data)
      val dOut = gradOutput.data

      for {
        n <- 0 until batchSize
        c <- 0 until channels

        h <- 0 until hPrime
        h1 = h * stride
        h2 = h1 + poolHeight

        w <- 0 until wPrime
        w1 = w * stride
        w2 = w1 + poolWidth
      } {
        val window = x.data(n, c, h1 :> h2, w1 :> w2)
        val upd = (window == out(n, c, h, w)) * dOut(n, c, h, w)
        dx(n, c, h1 :> h2, w1 :> w2) := upd
      }

      x.backward(Variable(dx))

      val bwEnd = System.currentTimeMillis()
      logger.debug(s"backward pass took ${bwEnd - bwStart} ms.")

      /*
big fat bug in broadcast of nd4j ndarrays, so cannot use below

val dxReshaped = ns.zerosLike(xReshaped)
val outShape = out.shape
val outNewAxis = out.reshape(outShape.head, outShape(1), outShape(2), 1, outShape(3), 1)
val mask: Tensor = xReshaped == outNewAxis
val dOutShape = dOut.shape.patch(3, Seq(1), 0) :+ 1
val dOutNewAxis = dOut.reshape(dOutShape)
val dOutBroadcast = new Tensor(
  ns.Ops.broadcastArrays(Seq(dOutNewAxis.array, dxReshaped.array)).head)
dxReshaped(mask) := dOutBroadcast(mask).asTensor
// dxReshaped /= ns.sum(ns.sum(mask, axis=5), axis=3)
val dx = dxReshaped.reshape(x.shape: _*)
x.backward(Variable(dx))
     */
    }
  }

  case class NaiveMaxPool2dFunction(x: Variable,
                                    poolHeight: Int,
                                    poolWidth: Int,
                                    stride: Int)
      extends Function {

    val List(batchSize, numChannels, hPrime, wPrime) =
      outputShape(x.shape, poolHeight, poolWidth, stride)

    val out: Tensor = ns.zeros(batchSize, numChannels, hPrime, wPrime)

    override def forward(): Variable = {

      for {
        n <- 0 until batchSize

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

// basically, this is
// dx = (x.data == out) * gradOutput.data
// but the shapes don't fit

      val dx = ns.zerosLike(x.data)
      val dOut = gradOutput.data

      for {
        n <- 0 until batchSize
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
        // If there are multiple argmaxes, this method will assign gradient to
        // ALL argmax elements of the input rather than picking one. In this case the
        // gradient will actually be incorrect. However this is unlikely to occur in
        // practice, so it shouldn't matter much.
        // val upd = (window == ns.max(window)) * dOut(n, c, h, w)
        val upd = (window == out(n, c, h, w)) * dOut(n, c, h, w)
        dx(n, c, h1 :> h2, w1 :> w2) := upd
      }

      x.backward(Variable(dx))
    }
  }
}
