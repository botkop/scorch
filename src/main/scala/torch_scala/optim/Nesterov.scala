package torch_scala.optim

import torch_scala.api.aten.{Tensor, TensorType}
import torch_scala.autograd.Variable

case class Nesterov[TT <: TensorType](parameters: Seq[Variable[Any, TT]], lr: Double, beta: Double = 0.9)
    extends Optimizer(parameters) {

  val vs: Seq[Tensor[Any, TT]] = parameters.map(p => Tensor.zeros_like(p.data))

  override def step(): Unit = parameters.zip(vs).foreach {
    case (p, v) =>
      val vPrev = v.to(v)
      v *= beta
      v -= p.grad.data * lr
      p.data += vPrev * (-beta) + v * (1 + beta)
  }
}
