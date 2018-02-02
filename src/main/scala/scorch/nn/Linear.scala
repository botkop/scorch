package scorch.nn

import botkop.{numsca => ns}
import botkop.numsca.Tensor
import scorch.autograd.Variable

case class Linear(weights: Variable, bias: Variable)
    extends Module(Seq(weights, bias)) {
  override def forward(x: Variable): Variable = {
    x.dot(weights.t()) + bias
  }
}

object Linear {
  def apply(inFeatures: Int, outFeatures: Int): Linear = {
    val w: Tensor =
      ns.randn(outFeatures, inFeatures) * math.sqrt(2.0 / outFeatures)
    val weights = Variable(w)
    val b: Tensor = ns.zeros(1, outFeatures)
    val bias = Variable(b)
    Linear(weights, bias)
  }
}
