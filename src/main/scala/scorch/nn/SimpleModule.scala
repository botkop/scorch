package scorch.nn

import com.typesafe.scalalogging.LazyLogging
import scorch.autograd.{DropoutFunction, Threshold, Variable}

abstract class SimpleModule(localParameters: Seq[Variable] = Nil)
    extends Module with LazyLogging {
  def forward(x: Variable): Variable = super.forward(Seq(x)).head
  def apply(x: Variable): Variable = forward(x)
}

/**
  * Extension of Module, that allows to pass multiple variables in the forward pass
  * This forward pass also returns multiple variables.
  * The single parameter forward pass is deactivated.
  * @param localParameters the trainable parameters of this module
  */
abstract class Module(localParameters: Seq[Variable] = Nil) {
  def forward(xs: Seq[Variable]): Seq[Variable]
  def apply(xs: Variable*): Seq[Variable] = forward(xs)

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

  def subModules: Seq[SimpleModule] = Seq.empty
  def parameters: Seq[Variable] =
    localParameters ++ subModules.flatMap(_.parameters)

  def zeroGrad(): Unit =
    parameters.map(_.grad).foreach(g => g.data := 0)
}

// proposal for a new structure
/*
abstract class SimpleModule(localParameters: Seq[Variable]) extends MultiVarModule(localParameters = localParameters)
{
  override def forward(x: Variable): Variable = super.forward(Seq(x)).head
}
*/




case class Relu() extends SimpleModule {
  override def forward(x: Variable): Variable = Threshold(x, 0).forward()
}

case class Dropout(p: Double = 0.5) extends SimpleModule {
  override def forward(x: Variable): Variable =
    DropoutFunction(x, p, inTrainingMode).forward()
}
