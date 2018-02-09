package scorch

import scorch.autograd.{SoftmaxLoss, Variable}

package object nn {
  def relu(x: Variable): Variable = Relu().forward(x)
  def softmax(x: Variable, y: Variable): Variable = SoftmaxLoss(x, y).forward()
}
