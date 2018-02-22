package scorch

import scorch.autograd.{SoftmaxLoss, Variable}
import scorch.examples.DinosaurIslandCharRnn.CrossEntropyLoss

package object nn {
  def relu(x: Variable): Variable = Relu().forward(x)
  def softmaxLoss(x: Variable, y: Variable): Variable =
    SoftmaxLoss(x, y).forward()

  /**
    * Convenience method for computing the loss.
    * Instantiates a CrossEntropyLoss object, and applies it
    * @param actuals source for the loss function
    * @param targets targets to compute the loss against
    * @return the loss variable, which can be backpropped into
    */
  def crossEntropyLoss(actuals: Seq[Variable], targets: Seq[Int]): Variable =
    CrossEntropyLoss(actuals, targets).forward()

}
