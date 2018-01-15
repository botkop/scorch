package scorch.nn

import botkop.numsca.Tensor
import scorch.autograd.Variable
import botkop.{numsca => ns}

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

case class Nesterov(parameters: Seq[Variable],
                    lr: Double,
                    beta: Double = 0.9)
    extends Optimizer(parameters) {

  val vs: Seq[Tensor] = parameters.map(p => ns.zeros(p.shape: _*))

  override def step(): Unit = parameters.zip(vs).foreach {
    case (p, v) =>
      val vPrev = v.copy()
      v *= beta
      v -= lr * p.grad.get.data
      p.data += (-beta * vPrev) + (1 + beta) * v
  }
}
