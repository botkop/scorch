package botkop.autograd

import botkop.{numsca => ns}
import botkop.numsca.Tensor

case class Variable(data: Tensor, gradFn: Option[Function] = None) {
  var g: Option[Tensor] = None
  def grad: Option[Variable] = g.map(Variable(_))

  def backward(gradOutput: Variable = Variable(ns.ones(data.shape))): Unit = {
    if (g.isEmpty) g = Some(ns.zeros(gradOutput.data.shape))
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

}

object Variable {
  def apply(d: Double): Variable = Variable(Tensor(d))
}
