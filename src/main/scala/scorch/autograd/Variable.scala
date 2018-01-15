package scorch.autograd

import botkop.{numsca => ns}
import botkop.numsca.Tensor
import com.typesafe.scalalogging.LazyLogging

case class Variable(data: Tensor, gradFn: Option[Function] = None)
    extends LazyLogging {

  var g: Option[Tensor] = None
  def grad: Option[Variable] = g.map(Variable(_))
  def shape: List[Int] = data.shape.toList

  def backward(): Unit = {
    logger.debug(s"no gradient output passed, initializing shape $shape")
    backward(Variable(ns.ones(data.shape)))
  }

  def backward(gradOutput: Variable): Unit = {
    logger.debug(s"gradient output shape: ${gradOutput.shape}")

    if (g.isEmpty) {
      g = Some(ns.zeros(gradOutput.data.shape))
      logger.debug(
        s"g not yet initialized, setting shape ${g.get.shape.toList}")
    }

    if (!g.get.sameShape(data)) {
      logger.warn(
        s"g and data have different shapes: g: ${g.get.shape.toList}, data: ${data.shape.toList}")
    }

    g.get += gradOutput.data
    for (gf <- gradFn) gf.backward(gradOutput)
  }

  def +(other: Variable): Variable = Add(this, other).forward()
  def -(other: Variable): Variable = Sub(this, other).forward()
  def *(other: Variable): Variable = Mul(this, other).forward()
  def /(other: Variable): Variable = Div(this, other).forward()
  def **(other: Variable): Variable = Pow(this, other).forward()

  def dot(other: Variable): Variable = Dot(this, other).forward()

  def unary_- : Variable = Negate(this).forward()
  def +(d: Double): Variable = AddConstant(this, d).forward()
  def -(d: Double): Variable = SubConstant(this, d).forward()
  def *(d: Double): Variable = MulConstant(this, d).forward()
  def /(d: Double): Variable = DivConstant(this, d).forward()
  def **(d: Double): Variable = PowConstant(this, d).forward()

  def mean(): Variable = Mean(this).forward()
  def threshold(d: Double): Variable = Threshold(this, d).forward()
  def t(): Variable = Transpose(this).forward()

}

object Variable {
  def apply(d: Double): Variable = Variable(Tensor(d))
}
