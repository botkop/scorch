package scorch.autograd

import botkop.{numsca => ns}
import botkop.numsca.Tensor
import com.typesafe.scalalogging.LazyLogging

object Variable {
  def apply(d: Double): Variable = Variable(Tensor(d))
  def apply(d: Double, name: Option[String]): Variable =
    Variable(Tensor(d), name = name)
}

case class Variable(data: Tensor,
                    gradFn: Option[Function] = None,
                    name: Option[String] = None)
    extends LazyLogging {

  override def toString: String =
    if (name.isDefined) s"name: ${name.get}, data: $data" else s"data: $data"

  lazy val grad: Variable =
    Variable(ns.zerosLike(data), name = name.map(n => s"g_$n"))
  def shape: List[Int] = data.shape.toList

  def backward(): Unit = {
    backward(Variable(ns.ones(data.shape)))
  }

  def backward(gradOutput: Variable): Unit = {
    grad.data += gradOutput.data
    for (gf <- gradFn) gf.backward(gradOutput)
  }

  def detach(name: Option[String] = None) = Variable(data, name = name)

  def +(other: Variable): Variable = Add(this, other).forward()
  def -(other: Variable): Variable = Sub(this, other).forward()
  def *(other: Variable): Variable = Mul(this, other).forward()
  def /(other: Variable): Variable = Div(this, other).forward()

  def dot(other: Variable): Variable = Dot(this, other).forward()

  def unary_- : Variable = Negate(this).forward()
  def +(d: Double): Variable = AddConstant(this, d).forward()
  def -(d: Double): Variable = SubConstant(this, d).forward()
  def *(d: Double): Variable = MulConstant(this, d).forward()
  def /(d: Double): Variable = DivConstant(this, d).forward()
  def **(d: Double): Variable = PowConstant(this, d).forward()

  def t(): Variable = Transpose(this).forward()
  def reshape(shape: List[Int]): Variable = Reshape(this, shape).forward()
  def reshape(shape: Int*): Variable = reshape(shape.toList)

  def exp(): Variable = Exp(this).forward()
  def mean(): Variable = Mean(this).forward()
  def mean(axis: Int): Variable = MeanByAxis(this, axis).forward()
  def sigmoid(): Variable = Sigmoid(this).forward()
  def softmax(): Variable = Softmax(this).forward()
  def tanh(): Variable = Tanh(this).forward()
  def relu(): Variable = Threshold(this, 0).forward()
  def variance(): Variable = Variance(this).forward()
  def variance(axis: Int): Variable = VarianceByAxis(this, axis).forward()
  def sqrt(): Variable = Sqrt(this).forward()
  def abs(): Variable = Abs(this).forward()

  def cat(w: Variable, axis: Int = 0): Variable =
    Concat(this, w, axis).forward()

}
