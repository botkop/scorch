package torch_scala.optim

import torch_scala.api.aten.TensorType
import torch_scala.autograd.Variable

abstract class Optimizer[TT <: TensorType](parameters: Seq[Variable[Any, TT]]) {
  def step(): Unit
  def zeroGrad(): Unit = parameters.foreach(_.zero_grad())
}

