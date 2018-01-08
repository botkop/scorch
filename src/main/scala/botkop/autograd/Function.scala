package botkop.autograd

import botkop.numsca.Tensor
import botkop.{numsca => ns}

trait Function {
  def forward(): Variable
  def backward(gradOutput: Variable): Unit
}

case class Add(v1: Variable, v2: Variable) extends Function {
  def forward(): Variable = Variable(v1.data + v2.data, gradFn = Some(this))
  def backward(gradOutput: Variable): Unit = {
    v2.backward(gradOutput)
    v1.backward(gradOutput)
  }
}

case class AddConstant(v: Variable, d: Double) extends Function {
  override def forward(): Variable = Variable(v.data + d, Some(this))
  override def backward(gradOutput: Variable): Unit =
    v.backward(gradOutput)
}

case class Sub(v1: Variable, v2: Variable) extends Function {
  def forward(): Variable = Variable(v1.data - v2.data, gradFn = Some(this))
  def backward(gradOutput: Variable): Unit = {
    v1.backward(gradOutput)
    v2.backward(Variable(-gradOutput.data))
  }
}

case class SubConstant(v: Variable, d: Double) extends Function {
  override def forward(): Variable = Variable(v.data + d, Some(this))
  override def backward(gradOutput: Variable): Unit =
    v.backward(gradOutput)
}

case class Mul(v1: Variable, v2: Variable) extends Function {
  override def forward(): Variable = Variable(v1.data * v2.data, Some(this))
  override def backward(gradOutput: Variable): Unit = {
    v1.backward(Variable(v2.data * gradOutput.data))
    v2.backward(Variable(v1.data * gradOutput.data))
  }
}

case class MulConstant(v: Variable, d: Double) extends Function {
  override def forward(): Variable = Variable(v.data * d, Some(this))
  override def backward(gradOutput: Variable): Unit = {
    v.backward(Variable(gradOutput.data * d))
  }
}

case class Div(v1: Variable, v2: Variable) extends Function {
  override def forward(): Variable = Variable(v1.data / v2.data, Some(this))
  override def backward(gradOutput: Variable): Unit = {
    val rv2 = 1 / v2.data
    val gv1 = gradOutput.data * rv2
    val gv2 = -gradOutput.data * v1.data * (rv2 ** 2)

    v1.backward(Variable(gv1))
    v2.backward(Variable(gv2))
  }
}

case class DivConstant(v: Variable, d: Double) extends Function {
  override def forward(): Variable = Variable(v.data / d, Some(this))
  override def backward(gradOutput: Variable): Unit = {
    v.backward(Variable(gradOutput.data / d))
  }
}

case class Pow(a: Variable, b: Variable) extends Function {
  override def forward(): Variable = Variable(a.data ** b.data, Some(this))
  override def backward(gradOutput: Variable): Unit = {
    val ga = gradOutput.data * b.data * (a.data ** (b.data - 1))
    val gb = gradOutput.data * (a.data ** b.data) * ns.log(a.data)
    a.backward(Variable(ga))
    b.backward(Variable(gb))
  }
}

case class PowConstant(v: Variable, d: Double) extends Function {
  override def forward(): Variable = Variable(ns.power(v.data, d), Some(this))
  val cache: Tensor = d * ns.power(v.data, d - 1)
  override def backward(gradOutput: Variable): Unit =
    v.backward(Variable(cache * gradOutput.data))
}

case class Negate(v: Variable) extends Function {
  override def forward(): Variable = Variable(-v.data, Some(this))
  override def backward(gradOutput: Variable): Unit =
    v.backward(Variable(-gradOutput.data))
}

case class Dot(v1: Variable, v2: Variable) extends Function {
  val w: Tensor = v1.data
  val x: Tensor = v2.data

  override def forward(): Variable = Variable(w dot x, Some(this))
  override def backward(gradOutput: Variable): Unit = {
    val dd = gradOutput.data
    val dw = dd dot x.T
    val dx = w.T dot dd

    v1.backward(Variable(dw))
    v2.backward(Variable(dx))
  }
}

//===============================================================

case class Exp(v: Variable) extends Function {
  val cache: Tensor = ns.exp(v.data)
  def forward() = Variable(data = cache, gradFn = Some(this))
  def backward(gradOutput: Variable): Unit = {
    v.backward(Variable(gradOutput.data * cache))
  }
}

case class Mean(v: Variable) extends Function {
  def forward() = Variable(data = ns.mean(v.data), gradFn = Some(this))
  def backward(gradOutput: Variable): Unit = {
    val n = v.data.shape.product
    v.backward(Variable(gradOutput.data / n))
  }
}

case class Max(x: Variable, y: Variable) extends Function {
  def forward(): Variable = {
    val max: Tensor = ns.maximum(x.data, y.data)
    Variable(max, Some(this))
  }
  override def backward(gradOutput: Variable): Unit = {
    x.backward(Variable((x.data >= y.data) * gradOutput.data))
    y.backward(Variable((x.data <= y.data) * gradOutput.data))
  }
}

case class Threshold(x: Variable, d: Double) extends Function {
  override def forward(): Variable = Variable(ns.maximum(x.data, d), Some(this))

  override def backward(gradOutput: Variable): Unit = {
    x.backward(Variable(gradOutput.data * (x.data > d)))
  }
}
