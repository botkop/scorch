import scorch.autograd._

package object scorch {
  implicit class AutoGradDoubleOps(d: Double) {
    def +(v: Variable): Variable = v + d
    def -(v: Variable): Variable = -v + d
    def *(v: Variable): Variable = v * d
    def /(v: Variable): Variable = (v ** -1) * d
  }

  // single parameter functions
  def exp(v: Variable): Variable = v.exp()
  def mean(v: Variable): Variable = v.mean()
  def sigmoid(v: Variable): Variable = v.sigmoid()
  def softmax(v: Variable): Variable = v.softmax()
  def tanh(v: Variable): Variable = v.tanh()
  def relu(v: Variable): Variable = v.relu()

  // multi parameter functions
  def cat(v: Variable, w: Variable, axis: Int = 0): Variable =
    Concat(v, w, axis).forward()

  // loss functions
  def softmaxLoss(x: Variable, y: Variable): Variable =
    SoftmaxLoss(x, y).forward()
  /**
    * Instantiates a CrossEntropyLoss object, and applies it
    * @param actuals source for the loss function
    * @param targets targets (classes) to compute the loss against
    * @return the loss variable, which can be backpropped into
    */
  def crossEntropyLoss(actuals: Seq[Variable], targets: Seq[Int]): Variable =
    CrossEntropyLoss(actuals, targets).forward()

}
