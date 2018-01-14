package scorch

import scorch.autograd.{SoftMax, Variable}

package object nn {

  def relu(x: Variable): Variable = Relu().forward(x)

  def softmax(x: Variable, y: Variable): Variable = SoftMax(x, y).forward()

}
