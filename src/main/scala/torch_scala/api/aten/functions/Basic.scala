package torch_scala.api.aten.functions


import org.bytedeco.javacpp.PointerPointer
import org.bytedeco.javacpp.annotation._
import torch_scala.NativeLoader
import torch_scala.api.aten._

import scala.reflect.ClassTag

@Platform(include = Array("torch/all.h", "helper.h"))
@Namespace("at") @NoOffset object Basic extends NativeLoader {

  implicit class BasicTensor[T: ClassTag, TT <: TensorType](self: Tensor[T, TT]) {
    def split(size: Int, dim: Int): Array[Tensor[T, TT]] = Basic.split(self, size, dim).data()
    def split_with_sizes(sizes: Array[Int], dim: Int): Array[Tensor[T, TT]] = Basic.split_with_sizes(self, IntList(sizes), dim).data()
  }

  @native @ByVal private def concat[T, TT <: TensorType](@ByRef t1: Tensor[T, TT], @ByRef t2: Tensor[T, TT], @Cast(Array("long")) dim: Long): Tensor[T, TT]

  @native @ByVal private def concat[T, TT <: TensorType](@ByVal ts: TensorVector[T, TT], @Cast(Array("long")) dim: Long): Tensor[T, TT]


  def cat[T: ClassTag, TT <: TensorType](t1: Tensor[T, TT], t2: Tensor[T, TT], dim: Int): Tensor[T, TT] = {
    new Tensor[T, TT](concat(t1, t2, dim))
  }

  def cat[T: ClassTag, TT <: TensorType](ts: Array[Tensor[T, TT]], dim: Int): Tensor[T, TT] = {
    val tvec = new TensorVector[T, TT]()
    ts.foreach(t => tvec.push_back(t))
    new Tensor[T, TT](concat(tvec, dim))
  }

  def cat[T: ClassTag, TT <: TensorType](ts: Tensor[T, TT]*)(dim: Int): Tensor[T, TT] = cat(ts.toArray, dim)

  @native @ByVal def split[T, TT <: TensorType](@ByRef self: Tensor[T, TT],
                                                   @Cast(Array("long")) size: Long,
                                                   @Cast(Array("long")) dim: Long): TensorVector[T, TT]

  @native @ByVal def split_with_sizes[T, TT <: TensorType](@ByRef self: Tensor[T, TT],
                                                              @ByVal sizes: IntList,
                                                              @Cast(Array("long")) dim: Long): TensorVector[T, TT]

}
