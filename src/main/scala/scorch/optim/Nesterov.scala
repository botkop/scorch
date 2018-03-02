package scorch.optim

import botkop.{numsca => ns}
import botkop.numsca.Tensor
import scorch.autograd.Variable

case class Nesterov(parameters: Seq[Variable], lr: Double, beta: Double = 0.9)
    extends Optimizer(parameters) {

  val vs: Seq[Tensor] = parameters.map(p => ns.zeros(p.shape: _*))

  override def step(): Unit = parameters.zip(vs).foreach {
    case (p, v) =>
      val vPrev = v.copy()
      v *= beta
      v -= lr * p.grad.data
      p.data += (-beta * vPrev) + (1 + beta) * v
  }
}
