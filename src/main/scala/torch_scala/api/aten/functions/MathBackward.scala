package torch_scala.api.aten.functions

import org.bytedeco.javacpp.annotation._
import torch_scala.NativeLoader
import torch_scala.api.aten.{IntList, Scalar, Tensor, TensorType}
import torch_scala.api.types.{FloatOrDouble, IsFloatOrDouble}

import scala.reflect.ClassTag


@Platform(include = Array("torch/all.h"))
@Namespace("at") @NoOffset object MathBackward extends NativeLoader {

  @native @ByVal def tanh_backward[T, TT <: TensorType](@ByRef gradOutput: Tensor[T, TT], @ByRef output: Tensor[T, TT]): Tensor[T, TT]



}
