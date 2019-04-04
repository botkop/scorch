package torch_scala.optim

import torch_scala.api.aten.{Tensor, TensorOptions, TensorType}
import torch_scala.autograd.Variable
import torch_scala.api.aten.functions.Math._
import torch_scala.api.types.FloatOrDouble

import scala.reflect.ClassTag

case class Adam[TT <: TensorType](parameters: Seq[Variable[Any, TT]],
                                  lr: Double,
                                  beta1: Double = 0.9,
                                  beta2: Double = 0.999,
                                  epsilon: Double = 1e-8)
    extends Optimizer(parameters) {

  implicit val opt: TensorOptions[Double, TT] = TensorOptions[Double, TT](parameters.head.data.device())
  val ms: Seq[Tensor[Double, TT]] = parameters.map(p => Tensor.zeros[Double, TT](p.shape))
  val vs: Seq[Tensor[Double, TT]] = parameters.map(p => Tensor.zeros[Double, TT](p.shape))

  var t = 1

  override def step(): Unit = {
    parameters.zip(ms).zip(vs).foreach {
      case ((p, m), v) =>
        val x = p.data.cast[Double]
        val dx = p.grad.data.cast[Double]

        m *= beta1
        m += dx * (1 - beta1)
        val mt = m / (1 - math.pow(beta1, t))

        v *= beta2
        v += dx.**(2) * (1 - beta2)
        val vt = v / (1 - math.pow(beta2, t))

        x -= (mt * lr) / (vt.sqrt() + epsilon)

        p.data.set(x.to(p.data))
    }
    t += 1
  }
}
