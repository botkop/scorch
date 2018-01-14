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

}
