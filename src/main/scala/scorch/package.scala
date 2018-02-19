import scorch.autograd.{Concat, Variable}

package object scorch {
  def cat(v: Variable, w: Variable, axis: Int = 0): Variable = Concat(v, w).forward()
}
