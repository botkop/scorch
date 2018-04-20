package scorch.optim

import scorch.autograd.Variable

abstract class Optimizer(val parameters: Seq[Variable]) {
  def step(): Unit
  def zeroGrad(): Unit =
    parameters.map(_.grad).foreach(g => g.data := 0)
}

