package scorch.nn.cnn

import botkop.{numsca => ns}
import botkop.numsca._
import com.typesafe.scalalogging.LazyLogging
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.convolution.Convolution
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.factory.Nd4j.PadMode
import scorch.nn.Module
import scorch.autograd.{Function, Variable}

case class Conv2d(w: Variable, b: Variable, pad: Int, stride: Int)
    extends Module(Seq(w, b)) {

  import Conv2d._

  def outputShape(inputShape: List[Int]): List[Int] =
    Conv2d.outputShape(inputShape, w.shape, pad, stride)

  override def forward(x: Variable): Variable =
    Im2colConv2dFunction(x, w, b, pad, stride).forward()
  // NaiveConv2dFunction(x, w, b, pad, stride).forward()
}

object Conv2d extends LazyLogging {

  // todo: there is a bug when padding = 0

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
    val List(numFilters, _, filterHeight, filterWidth) = wShape
    val List(numSamples, _, height, width) = xShape
    val hPrime: Int = 1 + (height + 2 * pad - filterHeight) / stride
    val wPrime: Int = 1 + (width + 2 * pad - filterWidth) / stride
    List(numSamples, numFilters, hPrime, wPrime)
  }

  /*
  case class Im2colConv2dFunction(x: Variable,
                                  w: Variable,
                                  b: Variable,
                                  pad: Int,
                                  stride: Int)
      extends Function {

    val List(batchSize, numFilters, hPrime, wPrime) =
      outputShape(x.shape, w.shape, pad, stride)

    val List(kernelHeight, kernelWidth) = w.shape.takeRight(2)

    override def forward(): Variable = {

      val xCols: Tensor = new Tensor(
        Convolution.im2col(x.data.array,
                           kernelHeight,
                           kernelWidth,
                           stride,
                           stride,
                           pad,
                           pad,
                           false))

      println(x.shape)
      println(x)
      println()
      println(xCols.shape.toList)
      println(xCols)
      println("!!!!!!!!!!!!!!!!!")
      // println(xCols.transpose(1, 2, 3, 4, 5, 0).shape.toList)
      // println(w.shape.toList)

      val ws = w.data.reshape(w.shape.head, -1)
      // val xt = xCols.transpose(3, 4, 5, 0, 1, 2).reshape(ws.shape.last, -1)
      val xt = xCols.reshape(ws.shape.last, -1)

      val res = ws.dot(xt) + b.data.reshape(-1, 1)

      val out = res
        .reshape(w.shape.head, hPrime, wPrime, x.shape.head)
        .transpose(3, 0, 1, 2)

      Variable(out, Some(this))

    }

    override def backward(gradOutput: Variable): Unit = ???
  }
   */

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

      val fwdStart = System.currentTimeMillis()

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

      val fwdEnd = System.currentTimeMillis()
      logger.debug(s"forward pass took ${fwdEnd - fwdStart} ms.")

      Variable(out, Some(this))
    }

    override def backward(gradOutput: Variable): Unit = {

      val bwStart = System.currentTimeMillis()

      val dOut = gradOutput.data

      val dx = zerosLike(x.data)
      val dw = zerosLike(w.data)
      val db = zerosLike(b.data)

      /*
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
       */

      (0 until numDataPoints).foreach { n =>
        val dxPad = ns.pad(dx(n), padArea, PadMode.CONSTANT)
        val xPad = ns.pad(x.data(n), padArea, PadMode.CONSTANT)

        (0 until numFilters).foreach { f =>
          val wf = w.data(f)

          (0 until hPrime).foreach { hp =>
            val h1 = hp * stride
            val h2 = h1 + hh

            (0 until wPrime).foreach { wp =>
              val w1 = wp * stride
              val w2 = w1 + ww
              val d = dOut(n, f, hp, wp)
              dxPad(:>, h1 :> h2, w1 :> w2) += wf * d
              dw(f) += xPad(:>, h1 :> h2, w1 :> w2) * d
              db(f) += d

              dx(n) := dxPad(:>, 1 :> -1, 1 :> -1)
            }
          }
        }
      }

      x.backward(Variable(dx))
      w.backward(Variable(dw))
      b.backward(Variable(db))

      val bwEnd = System.currentTimeMillis()
      logger.debug(s"backward pass took ${bwEnd - bwStart} ms.")
    }
  }

  /**
    * @param x image matrix to be translated into columns, (C,H,W)
    * @param hh filter height
    * @param ww filter width
    * @param stride stride
    * @return col: (new_h*new_w,hh*ww*C) matrix, each column is a cube that will convolve with a filter
    *              new_h = (H-hh) // stride + 1, new_w = (W-ww) // stride + 1
    */
  def im2col(x: Tensor, hh: Int, ww: Int, stride: Int): Tensor = {

    val Array(c, h, w) = x.shape
    val newH = (h - hh) / stride + 1
    val newW = w - ww / stride + 1


    val col = ns.zeros(newH * newW, c * hh * ww)
    println(col.shape.toList)
    println(x.shape.toList)

    for {
      i <- 0 until newH
      j <- 0 until newW
    } {
      val patch =
        x(:>, (i * stride) :> (i * stride + hh), (j * stride) :> (j * stride + ww))


      println(col(i * newW + j).shape.toList)
      println(patch.shape.toList)

      col(i * newW + j) := patch.reshape(1, -1)
    }

    col
  }

  /**
    * Args:
    * mul: (h_prime*w_prime*w,F) matrix, each col should be reshaped to C*h_prime*w_prime when C>0, or h_prime*w_prime when C = 0
    * h_prime: reshaped filter height
    * w_prime: reshaped filter width
    * C: reshaped filter channel, if 0, reshape the filter to 2D, Otherwise reshape it to 3D
    * Returns:
    * if C == 0: (F,h_prime,w_prime) matrix
    * Otherwise: (F,C,h_prime,w_prime) matrix
    */
  def col2im(mul: Tensor, hPrime: Int, wPrime: Int, c: Int): Tensor = {

    val f = mul.shape(1)

    if (c == 1) {
      val out = ns.zeros(f, hPrime, wPrime)
      for (i <- 0 until f) {
        val col = mul(:>, i)
        out(i) := col.reshape(hPrime, wPrime)
      }
      out
    } else {
      val out = ns.zeros(f, c, hPrime, wPrime)
      for (i <- 0 until f) {
        val col = mul(:>, i)
        out(i) := col.reshape(c, hPrime, wPrime)
      }
      out
    }
  }

  case class Im2colConv2dFunction(x: Variable,
                                  w: Variable,
                                  b: Variable,
                                  pad: Int,
                                  stride: Int)
      extends Function {

    override def forward(): Variable = {
      val List(batchSize, c, height, width) = x.shape
      val List(f, _, hh, ww) = w.shape

      val hPrime = (height + 2 * pad - hh) / stride + 1
      val wPrime = (width + 2 * pad - ww) / stride + 1

      val padArea = Array(Array(0, 0), Array(pad, pad), Array(pad, pad))

      val out = ns.zeros(batchSize, f, hPrime, wPrime)

      for (imNum <- 0 until batchSize) {
        val im = x.data(imNum)
        val imPad = ns.pad(im, padArea, PadMode.CONSTANT)
        val imCol = im2col(imPad, hh, ww, stride)
        val filterCol = w.data.reshape(f, -1).T
        val mul = imCol.dot(filterCol.T) + b.data
        out(imNum) := col2im(mul, hPrime, wPrime, 1)
      }
      Variable(out, Some(this))
    }

    override def backward(gradOutput: Variable): Unit = ???
  }
}
