package botkop.nn

import botkop.autograd.Variable

abstract class Optimizer(parameters: Seq[Variable]) {
  def step(): Unit
  def zeroGrad(): Unit =
    parameters.flatMap(_.grad).foreach(g => g.data := 0)
}

case class SGD(parameters: Seq[Variable], lr: Double)
    extends Optimizer(parameters) {
  override def step(): Unit = {
    parameters.foreach { p =>
      p.data -= p.grad.get.data * lr
    }
  }
}
