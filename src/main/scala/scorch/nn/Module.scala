package scorch.nn

import scorch.autograd.Variable
import botkop.numsca.Tensor
import botkop.{numsca => ns}
import com.typesafe.scalalogging.LazyLogging

trait Module extends LazyLogging {

  /*
  Pytorch way of solving distinction between training and test mode is by using a mutable variable.
  Perhaps there is a better way.
   */
  var inTrainingMode: Boolean = false

  /*
  Sets the module in training mode.
  This has any effect only on modules such as Dropout or BatchNorm.
   */
  def train(mode: Boolean = true): Unit = {
    this.inTrainingMode = mode
    subModules().foreach(_.inTrainingMode = mode)
  }

  /*
  Sets the module in evaluation mode.
  This has any effect only on modules such as Dropout or BatchNorm.
   */
  def eval(): Unit = train(false)

  def forward(x: Variable): Variable
  def apply(x: Variable): Variable = forward(x)
  def subModules(): Seq[Module] = Seq.empty
  def parameters(): Seq[Variable] = subModules().flatMap(_.parameters())
}

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

case class Dropout(p: Double = 0.5) extends Module {
  override def forward(x: Variable): Variable = x.dropout(p, inTrainingMode)
}

