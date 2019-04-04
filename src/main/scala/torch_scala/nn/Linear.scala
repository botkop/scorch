package torch_scala.nn

import torch_scala.api.aten.{Shape, Tensor, TensorOptions, TensorType}
import torch_scala.api.types.{FloatOrDouble, IsFloatOrDouble}
import torch_scala.autograd.Variable
import torch_scala.autograd.MathVariable._

import scala.reflect.ClassTag

case class Linear[T <: Any : IsFloatOrDouble : ClassTag: Numeric, TT <: TensorType](weights: Variable[T, TT], bias: Variable[T, TT])
  extends Module[T, TT, T, T](Seq(weights, bias)) {

  def forward(x: Variable[T, TT]): Variable[T, TT] = {
    x.mm(weights) + bias
  }
}

object Linear {
  def apply[T : IsFloatOrDouble : ClassTag: Numeric, TT <: TensorType](inFeatures: Int, outFeatures: Int)(
    implicit opt: TensorOptions[T, TT]
  ): Linear[T, TT] = {
    val w: Tensor[T, TT] = Tensor.randn[T, TT](Shape(inFeatures, outFeatures)) * math.sqrt(2.0 / outFeatures).asInstanceOf[T]
    val weights = Variable(w)
    val b: Tensor[T, TT] = Tensor.zeros[T, TT](Shape(1, outFeatures))
    val bias = Variable(b)
    new Linear(weights, bias)
  }
}