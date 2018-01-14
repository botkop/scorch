package botkop.nn

import botkop.autograd.Variable
import botkop.numsca.Tensor
import botkop.{numsca => ns}
import com.typesafe.scalalogging.LazyLogging

trait Module extends LazyLogging {
  def forward(x: Variable): Variable
  def apply(x: Variable): Variable = forward(x)
  def subModules(): Seq[Module] = Seq.empty
  def parameters(): Seq[Variable] = subModules().flatMap(_.parameters())
}

/*
abstract class Module(subModules: Module*) {
  def parameters(): Seq[Variable] = subModules.flatMap(_.parameters())
  def forward(x: Variable): Variable
  def apply(x: Variable): Variable = forward(x)
}

abstract class AbstractModule(subModules: Seq[Module]) extends Module {
}

case class AAA(override val subModules: Seq[Module]) extends AbstractModule(subModules) {
  override def forward(x: Variable): Variable = ???
}
*/

case class Linear(inFeatures: Int, outFeatures: Int) extends Module {
  val w: Tensor = ns.randn(outFeatures, inFeatures) * math.sqrt(2.0 / outFeatures)
  val weights = Variable(w)
  val b: Tensor = ns.zeros(1, outFeatures)
  val bias = Variable(b)

  override def forward(x: Variable): Variable = {
    x.dot(weights.t()) + bias
  }
  override def parameters(): Seq[Variable] = Seq(weights, bias)
}

case class Relu() extends Module {
  override def forward(x: Variable): Variable = x.threshold(0)
}
