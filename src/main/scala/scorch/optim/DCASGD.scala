package scorch.optim

import botkop.numsca.Tensor
import botkop.{numsca => ns}
import scorch.autograd.Variable

/**
  * Asynchronous Stochastic Gradient Descent with Delay Compensation
  * https://arxiv.org/abs/1609.08326
  */
case class DCASGD(parameters: Seq[Variable],
                  lr: Double,
                  wd: Double = 0.0,
                  useMomentum: Boolean = false,
                  momentum: Double = 0.9,
                  lambda: Double = 0.04)
    extends Optimizer(parameters) {

  val previousWeights: Seq[Tensor] = parameters.map { v =>
    v.data.copy()
  }
  val momenta: Seq[Option[Tensor]] = parameters.map { v =>
    if (useMomentum) Some(ns.zerosLike(v.data)) else None
  }

  var t = 0

  override def step(): Unit = {
    t += 1

    parameters.indices.foreach { i =>
      val weight = parameters(i).data
      val grad = parameters(i).grad.data
      val previousWeight = previousWeights(i)
      val maybeMomentum = momenta(i)

      val upd =
        -lr * (grad + (wd * weight) + lambda * grad * grad * (weight - previousWeight))

      val mom = maybeMomentum match {
        case None => upd
        case Some(m) =>
          m *= momentum
          m += upd
          m
      }

      previousWeight := weight
      weight += mom
    }
  }
}
