package torch_scala

import com.typesafe.scalalogging.LazyLogging
import torch_scala.api.aten.{Tensor, TensorType}
import torch_scala.autograd.Variable

import scala.reflect.ClassTag

object TestUtil extends LazyLogging {

  def relError[T, TT <: TensorType](x: Tensor[T, TT], y: Tensor[T, TT]): Double =
    ((x - y).abs() / (x.abs() + y.abs() + 1e-8.asInstanceOf[T])).max.scalar().toDouble()

  /**
    * Evaluate a numeric gradient for a function that accepts an array and returns an array.
    */
  def evalNumericalGradientArray[T: ClassTag, TT <: TensorType](f: Tensor[T, TT] => Tensor[T, TT],
                                 x: Tensor[T, TT],
                                 df: Tensor[T, TT],
                                 h: T = 1e-5.asInstanceOf[T])(implicit num: Numeric[T]): Tensor[T, TT] = {
    val grad = Tensor.zeros_like(x)

    for (i <- 0 until x.num_elements.toInt) {
      val ht = Tensor.zeros_like(x)
      ht.put(i, h)
      val pos = f(x + ht)
      val neg = f(x - ht)
      val g = ((pos - neg) * df).sum() / num.times(h, num.fromInt(2))

      grad.put(i, g.scalar())
    }

    grad
  }

  /**
    a naive implementation of numerical gradient of f at x
    - f should be a function that takes a single argument
    - x is the point (array) to evaluate the gradient at
    */
  def evalNumericalGradient[T: ClassTag, TT <: TensorType](f: Tensor[T, TT] => Double,
                            x: Tensor[T, TT],
                            h: Double = 0.00001)(implicit num: Numeric[T]): Tensor[T, TT] = {
    val grad = Tensor.zeros_like(x)
    val data = x.data()

    for (i <- 0 until x.num_elements.toInt) {
      val ht = Tensor.zeros_like(x)
      ht.put(i, h.asInstanceOf[T])
      val pos = f(x + ht)
      val neg = f(x - ht)
      val g = (pos - neg)  / (2 * h)

      grad.put(i, g.asInstanceOf[T])
    }

    grad
  }

  def binOpGradientCheck[T: ClassTag: Numeric, TT <: TensorType](f: (Variable[T, TT], Variable[T, TT]) => Variable[T, TT],
                         a: Variable[T, TT],
                         b: Variable[T, TT],
                         dOutOpt: Option[Variable[T, TT]] = None): Unit = {

    val out = f(a, b)
    logger.debug(s"out = $out")

    val dOut = dOutOpt.getOrElse(Variable[T, TT](Tensor.randn_like[T, TT](out.data).abs()))
    logger.debug(s"dOut = $dOut")

    out.backward(dOut)

    val da = a.grad.data
    val db = b.grad.data

    logger.debug(s"da = $da")
    logger.debug(s"db = $db")

    def fa(t: Tensor[T, TT]) = f(Variable[T, TT](t), b).data
    def fb(t: Tensor[T, TT]) = f(a, Variable[T, TT](t)).data

    val daNum = evalNumericalGradientArray[T, TT](fa, a.data, dOut.data)
    val dbNum = evalNumericalGradientArray(fb, b.data, dOut.data)

    logger.debug(s"daNum = $daNum")
    logger.debug(s"dbNum = $dbNum")

    val daError = relError(da, daNum)
    val dbError = relError(db, dbNum)

    logger.debug(s"daError = $daError")
    logger.debug(s"dbError = $dbError")

    assert(daError < 1e-5)
    assert(dbError < 1e-5)
  }

  def varConstOpGradientCheck[T: ClassTag: Numeric, TT <: TensorType](f: (Variable[T, TT], Double) => Variable[T, TT],
                              a: Variable[T, TT],
                              b: Double): Unit = {
    val out = f(a, b)
    logger.debug(s"out = $out")

    val dOut = Variable[T, TT](Tensor.randn_like(out.data))
    logger.debug(s"dOut = $dOut")

    out.backward(dOut)
    val da = a.grad.data
    logger.debug(s"da = $da")

    def fa(t: Tensor[T, TT]): Tensor[T, TT] = f(Variable[T, TT](t), b).data
    val daNum = evalNumericalGradientArray[T, TT](fa, a.data, dOut.data)
    logger.debug(s"daNum = $daNum")

    val daError = relError(da, daNum)
    logger.debug(s"daError = $daError")

    assert(daError < 1e-5)
  }

  def oneOpGradientCheck[T: ClassTag: Numeric, TT <: TensorType](f: (Variable[T, TT]) => Variable[T, TT], a: Variable[T, TT], errorBound: Double = 1e-8): Unit = {

    val out = f(a)
    logger.debug(s"out = $out")

    val dOut = Variable[T, TT](Tensor.randn_like(out.data))
    logger.debug(s"dOut = $dOut")

    out.backward(dOut)

    val da = a.grad.data
    logger.debug(s"da = $da")

    def fa(t: Tensor[T, TT]) = f(Variable(t)).data

    val daNum = evalNumericalGradientArray[T, TT](fa, a.data, dOut.data)
    logger.debug(s"daNum = $daNum")

    val daError = relError(da, daNum)
    logger.debug(s"daError = $daError")

    assert(daError < errorBound)
  }

}
