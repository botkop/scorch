package botkop

import botkop.autograd.{SoftMax, Variable}

package object nn {

  def relu(x: Variable): Variable = Relu().forward(x)

  def softmax(x: Variable, y: Variable): Variable = SoftMax(x, y).forward()

}
