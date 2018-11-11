package scorch.optim

import botkop.{numsca => ns}
import botkop.numsca.Tensor
import scorch.autograd.Variable

/**
  * Asynchronous Stochastic Gradient Descent with Delay Compensation
  * with adaptive lambda
  * https://arxiv.org/abs/1609.08326
  */
class DCASGDa(parameters: Seq[Variable],
              lr: Double,
              momentum: Double = 0.95,
              lambda: Double = 2)
  extends Optimizer(parameters) {

  val previousWeights: Seq[Tensor] = parameters.map { v =>
    v.data.copy()
  }
  val meanSquare: Seq[Tensor] = parameters.map { v =>
    ns.zerosLike(v.data)
  }
  val epsilon = 1e-7

  override def step(): Unit =
    parameters.indices.foreach { i =>
      val grad = parameters(i).grad.data
      val weight = parameters(i).data
      val previousWeight = previousWeights(i)

      meanSquare(i) *= momentum
      meanSquare(i) += (1 - momentum) * grad * grad

      val upd = -lr * (grad + lambda / ns.sqrt(meanSquare(i) + epsilon) * grad * grad * (weight - previousWeight))

      previousWeight := weight
      weight += upd
    }
}
