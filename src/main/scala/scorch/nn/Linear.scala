package scorch.nn

import botkop.{numsca => ns}
import botkop.numsca.Tensor
import scorch.autograd.Variable

case class Linear(weights: Variable, bias: Option[Variable])
  extends Module(if (bias.isDefined) Seq(weights, bias.get) else Seq(weights)) {

  override def forward(x: Variable): Variable = {
    if (bias.isDefined)
      x.dot(weights.t()) + bias.get
    else
      x.dot(weights.t())
  }
}

object Linear {

  def apply(weights: Variable): Linear = Linear(weights, None)
  def apply(weights: Variable, bias: Variable): Linear = Linear(weights, Some(bias))

  def apply(inFeatures: Int, outFeatures: Int, useBias: Boolean = true): Linear = {
    val w: Tensor =
      ns.randn(outFeatures, inFeatures) * math.sqrt(2.0 / outFeatures)
    val weights = Variable(w)

    if (useBias) {
      val b: Tensor = ns.zeros(1, outFeatures)
      val bias = Variable(b)
      Linear(weights, Some(bias))
    } else {
      Linear(weights)
    }
  }
}
