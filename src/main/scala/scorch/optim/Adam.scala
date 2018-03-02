package scorch.optim

import botkop.{numsca => ns}
import botkop.numsca.Tensor
import scorch.autograd.Variable

case class Adam(parameters: Seq[Variable],
                lr: Double,
                beta1: Double = 0.9,
                beta2: Double = 0.999,
                epsilon: Double = 1e-8)
    extends Optimizer(parameters) {

  val ms: Seq[Tensor] = parameters.map(p => ns.zeros(p.shape: _*))
  val vs: Seq[Tensor] = parameters.map(p => ns.zeros(p.shape: _*))

  var t = 1

  override def step(): Unit = parameters.zip(ms).zip(vs).foreach {
    case ((p, m), v) =>
      val x = p.data
      val dx = p.grad.data

      m *= beta1
      m += (1 - beta1) * dx
      val mt = m / (1 - math.pow(beta1, t))

      v *= beta2
      v += (1 - beta2) * ns.square(dx)
      val vt = v / (1 - math.pow(beta2, t))

      x -= lr * mt / (ns.sqrt(vt) + epsilon)

      t += 1
  }
}
