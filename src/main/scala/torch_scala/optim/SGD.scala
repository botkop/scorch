package torch_scala.optim

import torch_scala.api.aten.TensorType
import torch_scala.autograd.Variable

case class SGD[TT <: TensorType](parameters: Seq[Variable[Any, TT]], lr: Double)
    extends Optimizer(parameters) {
  override def step(): Unit =
    parameters.foreach { p =>
      p.data -= p.grad.data * lr
    }
}
