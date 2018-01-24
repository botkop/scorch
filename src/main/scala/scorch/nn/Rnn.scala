package scorch.nn

import scorch.autograd
import scorch.autograd.Variable

case class Rnn() extends Module {
  override def forward(x: Variable): Variable = ???
}

case class RnnStep(inputSize: Int, hiddenSize: Int, numLayers: Int) extends autograd.Function {

  override def forward(): Variable = ???

  override def backward(gradOutput: Variable): Unit = ???
}
