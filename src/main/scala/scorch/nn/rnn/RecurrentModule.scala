package scorch.nn.rnn

import com.typesafe.scalalogging.LazyLogging
import scorch.autograd.Variable
import scorch.nn.Module

abstract class RecurrentModule(localParameters: Seq[Variable] = Nil)
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

  def forward(xs: Seq[Variable]): Seq[Variable]
  def apply(xs: Variable*): Seq[Variable] = forward(xs)
  def subModules: Seq[Module] = Seq.empty
  def parameters: Seq[Variable] =
    localParameters ++ subModules.flatMap(_.parameters())
  def zeroGrad(): Unit =
    parameters.map(_.grad).foreach(g => g.data := 0)
}
