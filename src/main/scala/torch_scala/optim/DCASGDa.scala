package torch_scala.optim

import torch_scala.api.aten.{Tensor, TensorType}
import torch_scala.autograd.Variable

/**
  * Asynchronous Stochastic Gradient Descent with Delay Compensation
  * with adaptive lambda
  * https://arxiv.org/abs/1609.08326
  */
case class DCASGDa[TT <: TensorType](parameters: Seq[Variable[Any, TT]],
              lr: Double,
              momentum: Double = 0.95,
              lambda: Double = 2)
  extends Optimizer[TT](parameters) {

  val previousWeights: Seq[Tensor[Any, TT]] = parameters.map { v =>
    v.data.to(v.data)
  }
  val meanSquare: Seq[Tensor[Any, TT]] = parameters.map { v =>
    Tensor.zeros_like(v.data)
  }
  val epsilon = 1e-7

  override def step(): Unit =
    parameters.indices.foreach { i =>
      val grad = parameters(i).grad.data
      val weight = parameters(i).data
      val previousWeight = previousWeights(i)

      meanSquare(i) *= momentum
      meanSquare(i) += grad * grad * (1 - momentum)

      val upd = (grad + (meanSquare(i) + epsilon).sqrt().**(-1) * lambda * grad * grad * (weight - previousWeight)) * (-lr)

      previousWeight.set(weight)
      weight += upd
    }
}
