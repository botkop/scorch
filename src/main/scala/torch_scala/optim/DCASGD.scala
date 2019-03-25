package torch_scala.optim

import torch_scala.api.aten.{Tensor, TensorType}
import torch_scala.autograd.Variable

/**
  * Asynchronous Stochastic Gradient Descent with Delay Compensation
  * https://arxiv.org/abs/1609.08326
  */
case class DCASGD[TT <: TensorType](parameters: Seq[Variable[Any, TT]],
                  lr: Double,
                  wd: Double = 0.0,
                  useMomentum: Boolean = false,
                  momentum: Double = 0.9,
                  lambda: Double = 0.04)
    extends Optimizer[TT](parameters) {

  val previousWeights: Seq[Tensor[Any, TT]] = parameters.map { v =>
    v.data.to(v.data)
  }
  val momenta: Seq[Option[Tensor[Any, TT]]] = parameters.map { v =>
    if (useMomentum) Some(Tensor.zeros_like(v.data)) else None
  }

  override def step(): Unit = {

    parameters.indices.foreach { i =>
      val weight = parameters(i).data
      val grad = parameters(i).grad.data
      val previousWeight = previousWeights(i)
      val maybeMomentum = momenta(i)

      val upd =
         (grad + (weight * wd) + grad * grad * (weight - previousWeight) * lambda) * (-lr)

      val mom = maybeMomentum match {
        case None => upd
        case Some(m) =>
          m *= momentum
          m += upd
          m
      }

      previousWeight.set(weight)
      weight += mom
    }
  }
}
