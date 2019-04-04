package torch_scala.autograd


import torch_scala.api.aten.functions.{Basic, MathBackward}
import torch_scala.api.aten.functions.Basic._
import torch_scala.api.aten._
import torch_scala.api._

import scala.language.postfixOps
import torch_scala.api.aten.functions.Math._
import torch_scala.api.{doubleToScalar, intToScalar}
import torch_scala.autograd.MathVariable._

import scala.reflect.ClassTag

abstract class Function[T: ClassTag, TT <: TensorType](val name: String = "") {
  def forward(): Variable[T, TT]
  def backward(gradOutput: Variable[T, TT]): Unit

  def unbroadcast(v: Variable[T, TT], oldShape: Shape): Variable[T, TT] = {
    unbroadcast(v.data, oldShape)
  }

  def unbroadcast(data: Tensor[T, TT], oldShape: Shape): Variable[T, TT] = {
    val t = data.shape.asArray.zipWithIndex.reverse.foldLeft(data) {
      case (d: Tensor[T, TT], (ni, i)) =>
        if (oldShape.rank <= i)
          d.sum(i)
        else if (oldShape(i) == ni)
          d
        else if (oldShape(i) == 1)
          d.sum(Array(i), keepdim = true)
        else
          throw new Exception(
            s"unable to unbroadcast shape ${data.shape} to $oldShape")
    }
    Variable(t)
  }
}

case class Add[T: ClassTag, TT <: TensorType](v1: Variable[T, TT], v2: Variable[T, TT]) extends Function[T, TT]("Add") {
  def forward(): Variable[T, TT] = Variable[T, TT](
    v1.data + v2.data,
    gradFn = Some(this),
    name = Some("(" + v1.name.getOrElse("") + " + " + v2.name.getOrElse("") + ")"))
  def backward(gradOutput: Variable[T, TT]): Unit = {
    v1.backward(unbroadcast(gradOutput, v1.shape))
    v2.backward(unbroadcast(gradOutput, v2.shape))
  }
}

case class AddConstant[T: ClassTag, TT <: TensorType](v: Variable[T, TT], d: Scalar[T]) extends Function[T, TT]("Add_c") {
  override def forward(): Variable[T, TT] = Variable[T, TT](v.data + d, Some(this))
  override def backward(gradOutput: Variable[T, TT]): Unit = {
    v.backward(gradOutput)
  }
}

case class Sub[T: ClassTag, TT <: TensorType](v1: Variable[T, TT], v2: Variable[T, TT]) extends Function[T, TT]("Sub") {
  def forward(): Variable[T, TT] = Variable[T, TT](v1.data - v2.data, gradFn = Some(this), name = Some("(" + v1.name + " - " + v2.name + ")"))
  def backward(gradOutput: Variable[T, TT]): Unit = {
    v1.backward(unbroadcast(gradOutput, v1.shape))
    v2.backward(unbroadcast(-gradOutput.data, v2.shape))
  }
}

case class SubConstant[T: ClassTag, TT <: TensorType](v: Variable[T, TT], d: Scalar[T]) extends Function[T, TT] {
  override def forward(): Variable[T, TT] = Variable[T, TT](v.data + d, Some(this))
  override def backward(gradOutput: Variable[T, TT]): Unit = {
    v.backward(gradOutput)
  }
}

case class Mul[T: ClassTag, TT <: TensorType](v1: Variable[T, TT], v2: Variable[T, TT]) extends Function[T, TT] {
  override def forward(): Variable[T, TT] = Variable[T, TT](
    v1.data * v2.data,
    Some(this),
    name = Some("(" + v1.name.getOrElse("") + " * " + v2.name.getOrElse("") + ")")
  )
  override def backward(gradOutput: Variable[T, TT]): Unit = {
    val dv1 = v2.data * gradOutput.data
    val vdv1 = unbroadcast(dv1, v1.shape)
    val dv2 = v1.data * gradOutput.data
    val vdv2 = unbroadcast(dv2, v2.shape)
    v1.backward(vdv1)
    v2.backward(vdv2)
  }
}

case class MulConstant[T: ClassTag, TT <: TensorType](v: Variable[T, TT], d: Scalar[T]) extends Function[T, TT] {
  override def forward(): Variable[T, TT] = Variable[T, TT](v.data * d, Some(this))
  override def backward(gradOutput: Variable[T, TT]): Unit = {
    val dv = gradOutput.data * d
    v.backward(Variable[T, TT](dv))
  }
}

case class Div[T: ClassTag, TT <: TensorType](v1: Variable[T, TT], v2: Variable[T, TT])(implicit num: Numeric[T]) extends Function[T, TT] {
  override def forward(): Variable[T, TT] = Variable[T, TT](v1.data / v2.data, Some(this))
  override def backward(gradOutput: Variable[T, TT]): Unit = {
    val rv2 = Tensor.ones_like(v2.data) / v2.data
    val gv1 = gradOutput.data * rv2
    val gv2 = -gradOutput.data * v1.data * (rv2 ** num.fromInt(2))

    val vgv1 = unbroadcast(gv1, v1.shape)
    val vgv2 = unbroadcast(gv2, v2.shape)
    v1.backward(vgv1)
    v2.backward(vgv2)
  }
}

case class DivConstant[T: ClassTag, TT <: TensorType](v: Variable[T, TT], d: Scalar[T]) extends Function[T, TT] {
  override def forward(): Variable[T, TT] = Variable[T, TT](v.data / d, Some(this))
  override def backward(gradOutput: Variable[T, TT]): Unit = {
    val dv = gradOutput.data / d
    v.backward(Variable[T, TT](dv))
  }
}

/* Pow of 2 tensors is currently not implemented in numsca
case class Pow(a: Variable[T, TT], b: Variable[T, TT]) extends Function[T, TT] {


  override def forward(): Variable[T, TT] = {
    Variable[T, TT](a.data ** b.data, Some(this))
  }

  override def backward(gradOutput: Variable[T, TT]): Unit = {
     val ga = gradOutput.data * b.data * (a.data ** (b.data - 1))
     val gb = gradOutput.data * (a.data ** b.data) * ns.log(a.data)

    val vga = unbroadcast(ga, a.shape)
    val vgb = unbroadcast(gb, b.shape)

    logger.debug(s"pow backward, ga.shape=${vga.shape}, gb.shape=${vgb.shape}")
    a.backward(vga)
    b.backward(vgb)
  }
}
 */

case class PowConstant[T: ClassTag, TT <: TensorType](v: Variable[T, TT], d: Scalar[T])(implicit num: Numeric[T]) extends Function[T, TT]("Pow") {
  override def forward(): Variable[T, TT] = Variable[T, TT](v.data.**(d), Some(this), name = Some(v.name + s".pow($d)"))
  val cache: Tensor[T, TT] = v.data.**(num.minus(d.getValue, num.fromInt(1))) * d
  override def backward(gradOutput: Variable[T, TT]): Unit = {
    val dv = cache * gradOutput.data
    v.backward(Variable[T, TT](dv))
  }
}

case class Sqrt[T: ClassTag, TT <: TensorType](v: Variable[T, TT])(implicit num: Numeric[T]) extends Function[T, TT] {
  val cache: Tensor[T, TT] = v.data.sqrt()
  override def forward(): Variable[T, TT] = Variable[T, TT](cache, Some(this))
  override def backward(gradOutput: Variable[T, TT]): Unit = {
    val dv = (num.one / (cache * num.fromInt(2))) * gradOutput.data
    v.backward(Variable(dv))
  }
}

case class Abs[T: ClassTag, TT <: TensorType](v: Variable[T, TT]) extends Function[T, TT] {
  val cache: Tensor[T, TT] = v.data.abs()
  override def forward(): Variable[T, TT] = Variable[T, TT](cache, Some(this))
  override def backward(gradOutput: Variable[T, TT]): Unit = {
    val dv = (v.data / cache) * gradOutput.data
    v.backward(Variable[T, TT](dv))
  }
}

case class Negate[T: ClassTag, TT <: TensorType](v: Variable[T, TT]) extends Function[T, TT] {
  override def forward(): Variable[T, TT] = Variable[T, TT](-v.data, Some(this))
  override def backward(gradOutput: Variable[T, TT]): Unit = {
    val dv = -gradOutput.data
    v.backward(Variable[T, TT](dv))
  }
}

case class Dot[T: ClassTag, TT <: TensorType](v1: Variable[T, TT], v2: Variable[T, TT]) extends Function[T, TT] {
  val w: Tensor[T, TT] = v1.data
  val x: Tensor[T, TT] = v2.data

  override def forward(): Variable[T, TT] = Variable[T, TT](w dot x, Some(this))
  override def backward(gradOutput: Variable[T, TT]): Unit = {
    val dd = gradOutput.data
    val dw = x * dd
    val dx = w * dd
    v1.backward(Variable[T, TT](dw))
    v2.backward(Variable[T, TT](dx))
  }
}

case class Matmul[T: ClassTag, TT <: TensorType](v1: Variable[T, TT], v2: Variable[T, TT]) extends Function[T, TT] {
  val w: Tensor[T, TT] = v1.data
  val x: Tensor[T, TT] = v2.data

  override def forward(): Variable[T, TT] = Variable[T, TT](w matmul x, Some(this))
  override def backward(gradOutput: Variable[T, TT]): Unit = {
    val dd = gradOutput.data
    val dw = dd matmul x.T
    val dx = w.T matmul dd
    v1.backward(Variable[T, TT](dw))
    v2.backward(Variable[T, TT](dx))
  }
}

case class Transpose[T: ClassTag, TT <: TensorType](v: Variable[T, TT]) extends Function[T, TT] {
  override def forward(): Variable[T, TT] = Variable[T, TT](v.data.T, Some(this))
  override def backward(gradOutput: Variable[T, TT]): Unit =
    v.backward(Variable[T, TT](gradOutput.data.T))
}

// todo test this
case class Reshape[T: ClassTag, TT <: TensorType](v: Variable[T, TT], shape: Shape) extends Function[T, TT] {
  val oldShape: Shape = v.shape
  override def forward(): Variable[T, TT] =
    Variable[T, TT](v.data.reshape(shape), Some(this))
  override def backward(gradOutput: Variable[T, TT]): Unit = {
    val dv = gradOutput.data.reshape(oldShape)
    v.backward(Variable[T, TT](dv))
  }
}

case class Concat[T: ClassTag, TT <: TensorType](v1: Variable[T, TT], v2: Variable[T, TT], axis: Int = 0) extends Function[T, TT] {

  require(axis == 0 || axis == 1, "axis must be either 0 or 1")

  override def forward(): Variable[T, TT] = {
    Variable[T, TT](Basic.cat(v1.data, v2.data, axis), Some(this))
  }

  override def backward(gradOutput: Variable[T, TT]): Unit =
    if (axis == 0) {
      val d = gradOutput.data
      val d12 = d.split_with_sizes(Array(v1.shape(axis), v2.shape(axis)), axis)
      val dv1 = d12(0)
      val dv2 = d12(1)
      v1.backward(Variable[T, TT](dv1))
      v2.backward(Variable[T, TT](dv2))
    } else {
      val dv1 = gradOutput.data(::, 0 :: v1.shape(axis))
      val dv2 = gradOutput.data(::, v1.shape(axis) :: v1.shape(axis) + v2.shape(axis))
      v1.backward(Variable[T, TT](dv1))
      v2.backward(Variable[T, TT](dv2))
    }
}

//===============================================================

case class Exp[T: ClassTag, TT <: TensorType](v: Variable[T, TT]) extends Function[T, TT] {
  val cache: Tensor[T, TT] = v.data.exp()
  def forward(): Variable[T, TT] = Variable[T, TT](data = cache, gradFn = Some(this))
  def backward(gradOutput: Variable[T, TT]): Unit = {
    v.backward(Variable[T, TT](gradOutput.data * cache))
  }
}

case class Cos[T: ClassTag, TT <: TensorType](v: Variable[T, TT]) extends Function[T, TT] {
  val cache: Tensor[T, TT] = v.data.sin() * v.data.dataType.cast(-1)
  def forward() = Variable[T, TT](data = v.data.cos(), gradFn = Some(this))
  def backward(gradOutput: Variable[T, TT]): Unit = {
    v.backward(Variable[T, TT](gradOutput.data * cache))
  }
}

case class Sin[T: ClassTag, TT <: TensorType](v: Variable[T, TT]) extends Function[T, TT] {
  val cache: Tensor[T, TT] = v.data.cos()
  def forward() = Variable[T, TT](data = v.data.sin(), gradFn = Some(this))
  def backward(gradOutput: Variable[T, TT]): Unit = {
    v.backward(Variable[T, TT](gradOutput.data * cache))
  }
}

case class Tanh[T: ClassTag, TT <: TensorType](v: Variable[T, TT]) extends Function[T, TT] {
  val cache: Tensor[T, TT] = v.data.tanh()
  override def forward(): Variable[T, TT] = Variable[T, TT](cache, Some(this))
  override def backward(gradOutput: Variable[T, TT]): Unit =
    v.backward(Variable[T, TT](MathBackward.tanh_backward(gradOutput.data, cache)))
    //v.backward(Variable[T, TT]((1 - cache.**(2)) * gradOutput.data))
}

case class Sigmoid[T: ClassTag, TT <: TensorType](v: Variable[T, TT])(implicit num: Numeric[T]) extends Function[T, TT] {
  lazy val sigmoid: Tensor[T, TT] = v.data.sigmoid()
  override def forward(): Variable[T, TT] = Variable[T, TT](sigmoid, Some(this))
  override def backward(gradOutput: Variable[T, TT]): Unit =
    v.backward(Variable[T, TT](gradOutput.data * sigmoid * (num.one - sigmoid)))
}

case class Softmax[T: ClassTag, TT <: TensorType](v: Variable[T, TT], dim: Long) extends Function[T, TT] {
  lazy val softmax: Tensor[T, TT] = v.data.softmax(dim)
  override def forward(): Variable[T, TT] = Variable[T, TT](softmax, Some(this))
  override def backward(gradOutput: Variable[T, TT]): Unit = {
    // from https://stackoverflow.com/questions/33541930/how-to-implement-the-softmax-derivative-independently-from-any-loss-Function[T, TT]
    val y = softmax
    val dy = gradOutput.data

    val dx = y * dy
    val s = dx.sum(Array(dim.toInt), true)
    dx -= y * s
    v.backward(Variable[T, TT](dx))
  }
}

case class Mean[T: ClassTag, TT <: TensorType](v: Variable[T, TT])(implicit num: Numeric[T]) extends Function[T, TT]("Mean") {
  def forward(): Variable[T, TT] = Variable[T, TT](data = v.data.mean(), gradFn = Some(this), name = Some(v.name + ".mean"))
  def backward(gradOutput: Variable[T, TT]): Unit = {
    val n: T = v.shape.numElements.asInstanceOf[T]
    v.backward(Variable[T, TT](gradOutput.data / n))
  }
}

case class MeanByAxis[T: ClassTag, TT <: TensorType](v: Variable[T, TT], axis: Int)(implicit num: Numeric[T]) extends Function[T, TT] {
  def forward(): Variable[T, TT] = Variable[T, TT](data = v.data.mean(Array(axis), true), gradFn = Some(this))
  def backward(gradOutput: Variable[T, TT]): Unit = {
    val n: T = num.fromInt(v.shape(axis))
    v.backward(Variable[T, TT](gradOutput.data / n))
  }
}

case class Variance[T: ClassTag, TT <: TensorType](v: Variable[T, TT])(implicit num: Numeric[T]) extends Function[T, TT]("Var") {
  override def forward(): Variable[T, TT] = (v - v.mean()).pow(num fromInt 2).mean()
  override def backward(gradOutput: Variable[T, TT]): Unit = {
    v.backward(gradOutput)
  }
}

case class VarianceByAxis[T: ClassTag, TT <: TensorType](v: Variable[T, TT], axis: Int)(implicit num: Numeric[T]) extends Function[T, TT] {
  override def forward(): Variable[T, TT] = ((v - v.mean(axis)) ** num.fromInt(2)).mean(axis)
  override def backward(gradOutput: Variable[T, TT]): Unit =
    v.backward(gradOutput)
}

case class Max[T: ClassTag, TT <: TensorType](x: Variable[T, TT], y: Variable[T, TT]) extends Function[T, TT] {
  def forward(): Variable[T, TT] = {
    Variable[T, TT](x.data.maximum(y.data), Some(this))
  }
  override def backward(gradOutput: Variable[T, TT]): Unit = {
    x.backward(Variable[T, TT]((x.data ge y.data).to(x.data) * gradOutput.data))
    y.backward(Variable[T, TT]((x.data le y.data).to(x.data) * gradOutput.data))
  }
}

case class Threshold[T: ClassTag, TT <: TensorType](x: Variable[T, TT], d: Scalar[T]) extends Function[T, TT] {
  override def forward(): Variable[T, TT] = Variable(x.data.threshold(d, d), Some(this))
  override def backward(gradOutput: Variable[T, TT]): Unit = {
    x.backward(Variable[T, TT](gradOutput.data * (x.data gt d).to(x.data)))
  }
}

//============================================
// Loss Function[T, TT]s
case class SoftmaxLoss[T: ClassTag, TT <: TensorType](actual: Variable[T, TT], target: Variable[Long, TT])(implicit num: Numeric[T]) extends Function[T, TT] {
  val x: Tensor[T, TT] = actual.data
  val n: Int = x.shape(0)
  implicit val long_opt: TensorOptions[Long, TT] = x.long_options()
  val y: Tensor[Long, TT] = target.data.reshape(Shape(n))

  val shiftedLogits: Tensor[T, TT] = x - x.maximum(1, keepdim = true)
  val z: Tensor[T, TT] = shiftedLogits.exp().sum(Array(1), true)
  val logProbs: Tensor[T, TT] = shiftedLogits - z.log()
  val p: Int = x.shape(1)
  val m = Tensor.arange[Long, TT](0, n) * p + y
  val loss = - logProbs.take(m).sum() / num.fromInt(n)


  override def forward(): Variable[T, TT] = Variable[T, TT](loss, Some(this))

  override def backward(gradOutput: Variable[T, TT] = null /* not used */ ): Unit = {
    val dx: Tensor[T, TT] = logProbs.exp()
    val m = Tensor.arange[Long, TT](0, n) * p + y
    dx.put(m, dx.take(m) - num.one)

    actual.backward(Variable[T, TT](dx / num.fromInt(n)))
  }
}

///**
//  * Computes the cross-entropy loss
//  * @param actuals sequence of yHat Variable[T, TT]s
//  * @param targets sequence of Y indices (ground truth)
//  */
//case class CrossEntropyLoss[T: ClassTag, TT <: TensorType](actuals: Seq[Variable[T, TT]], targets: Seq[Int])
//    extends Function[T, TT] {
//
//  /**
//    * Computes the cross entropy loss, and wraps it in a Variable[T, TT].
//    * The Variable[T, TT] can be back propped into, to compute the gradients of the parameters
//    * @return the cross entropy loss Variable[T, TT]
//    */
//  override def forward(): Variable[T, TT] = {
//    val seqLoss = actuals.zip(targets).foldLeft(0.0) {
//      case (loss, (yht, y)) =>
//        loss - ns.log(yht.data(y, 0)).squeeze()
//    }
//    Variable[T, TT](Tensor(seqLoss), Some(this))
//  }
//
//  /**
//    * Compute the loss of each generated character, and back prop from last to first
//    * @param gradOutput not used
//    */
//  override def backward(gradOutput: Variable[T, TT]): Unit = {
//    actuals.zip(targets).reverse.foreach {
//      case (yh, y) =>
//        val dy = ns.copy(yh.data)
//        dy(y, 0) -= 1
//        yh.backward(Variable[T, TT](dy))
//    }
//  }
//}
