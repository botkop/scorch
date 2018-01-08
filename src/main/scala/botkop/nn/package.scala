package botkop

import botkop.autograd.Variable

package object nn {

  def relu(x: Variable): Variable = Relu().forward(x)

}
