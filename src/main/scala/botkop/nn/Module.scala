package botkop.nn

import botkop.autograd.Variable
import botkop.numsca.Tensor
import botkop.{numsca => ns}

trait Module {
  def forward(x: Variable): Variable
  def apply(x: Variable): Variable = forward(x)
}

case class Linear(inFeatures: Int, outFeatures: Int) extends Module {
  val w: Tensor = ns.randn(outFeatures, inFeatures) * math.sqrt(2.0 / outFeatures)
  val weights = Variable(w)
  val b: Tensor = ns.zeros(outFeatures, 1)
  val bias = Variable(b)

  override def forward(x: Variable): Variable = weights.dot(x) + bias
}

case class Relu() extends Module {
  override def forward(x: Variable): Variable = x.threshold(0)
}
