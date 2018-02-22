package scorch.nn

import com.typesafe.scalalogging.LazyLogging
import scorch.autograd.{DropoutFunction, Threshold, Variable}

abstract class Module(localParameters: Seq[Variable] = Nil)
    extends LazyLogging {

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
    subModules.foreach(_.train(mode))
  }

  /*
  Sets the module in evaluation mode.
  This has any effect only on modules such as Dropout or BatchNorm.
   */
  def eval(): Unit = train(false)

  def forward(x: Variable): Variable
  def apply(x: Variable): Variable = forward(x)
  def subModules: Seq[Module] = Seq.empty
  def parameters: Seq[Variable] =
    localParameters ++ subModules.flatMap(_.parameters)

  def zeroGrad(): Unit =
    parameters.map(_.grad).foreach(g => g.data := 0)
}

abstract class MultiVarModule(localParameters: Seq[Variable] = Nil)
    extends Module(localParameters) {
  override def forward(x: Variable): Variable =
    throw new UnsupportedOperationException("Use forward(xs: Seq[Variable])")
  def forward(xs: Seq[Variable]): Seq[Variable]
  def apply(xs: Variable*): Seq[Variable] = forward(xs)
}

case class Relu() extends Module {
  override def forward(x: Variable): Variable = Threshold(x, 0).forward()
}

case class Dropout(p: Double = 0.5) extends Module {
  override def forward(x: Variable): Variable =
    DropoutFunction(x, p, inTrainingMode).forward()
}
