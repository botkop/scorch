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

      val out = res.reshape(w.shape.head, hPrime, wPrime, x.shape.head).transpose(3, 0, 1, 2)

      Variable(out, Some(this))


    }

    override def backward(gradOutput: Variable): Unit = ???
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
}
