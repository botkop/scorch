package scorch

package object autograd {

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

}
