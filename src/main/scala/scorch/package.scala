import scorch.autograd._
import scorch.nn.Relu


package object scorch {
  implicit class AutoGradDoubleOps(d: Double) {
    def +(v: Variable): Variable = v + d
    def -(v: Variable): Variable = -v + d
    def *(v: Variable): Variable = v * d
    def /(v: Variable): Variable = (v ** -1) * d
  }

  def exp(v: Variable): Variable = Exp(v).forward()
  def mean(v: Variable): Variable = Mean(v).forward()
  def sigmoid(v: Variable): Variable = Sigmoid(v).forward()
  def softmax(v: Variable): Variable = Softmax(v).forward()
  def tanh(v: Variable): Variable = Tanh(v).forward()
  def cat(v: Variable, w: Variable, axis: Int = 0): Variable = Concat(v, w).forward()
  def relu(x: Variable): Variable = Threshold(x, 0).forward()
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
