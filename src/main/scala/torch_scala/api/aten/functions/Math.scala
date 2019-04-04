package torch_scala.api.aten.functions

import org.bytedeco.javacpp.annotation._
import torch_scala.NativeLoader
import torch_scala.api.aten.{IntList, Scalar, Tensor, TensorType}
import torch_scala.api.types.{FloatOrDouble, IsFloatOrDouble}

import scala.reflect.ClassTag


@Platform(include = Array("torch/all.h", "helper.h"))
@Namespace("at") @NoOffset object Math extends NativeLoader {

  @native @ByVal def cos[T, TT <: TensorType](@ByRef self: Tensor[T, TT]): Tensor[T, TT]
  @native @ByVal def sin[T, TT <: TensorType](@ByRef self: Tensor[T, TT]): Tensor[T, TT]
  @native @ByVal def exp[T, TT <: TensorType](@ByRef self: Tensor[T, TT]): Tensor[T, TT]
  @native @ByVal def tanh[T, TT <: TensorType](@ByRef self: Tensor[T, TT]): Tensor[T, TT]
  @native @ByVal def sigmoid[T, TT <: TensorType](@ByRef self: Tensor[T, TT]): Tensor[T, TT]
  @native @ByVal def log[T, TT <: TensorType](@ByRef self: Tensor[T, TT]): Tensor[T, TT]

  @native @ByVal def mean[T, TT <: TensorType](@ByRef self: Tensor[T, TT]): Tensor[T, TT]
  @native @ByVal def mean[T, TT <: TensorType](@ByRef self: Tensor[T, TT], @ByVal dim: IntList, keepdim: Boolean): Tensor[T, TT]
  @native @ByVal def maximum[T, TT <: TensorType](@ByRef self: Tensor[T, TT], @Cast(Array("long")) dim: Long, keepdim: Boolean): Tensor[T, TT]
  @native @ByVal def argmax[T, TT <: TensorType](@ByRef self: Tensor[T, TT], @Cast(Array("long")) dim: Long, keepdim: Boolean): Tensor[Long, TT]

  // @native @ByVal def `var`[T, TT <: TensorType](@ByRef self: Tensor[T, TT], unbiased: Boolean): Tensor[T, TT]
  // @native @ByVal def `var`[T, TT <: TensorType](@ByRef self: Tensor[T, TT], @ByVal dim: IntList, unbiased: Boolean, keepdim: Boolean): Tensor[T, TT]

  @native @ByVal def softmax[T, TT <: TensorType](@ByRef self: Tensor[T, TT], @Cast(Array("long")) dim: Long): Tensor[T, TT]

  @native @ByVal def matmul[T, TT <: TensorType](@ByRef self: Tensor[T, TT], @ByRef other: Tensor[T, TT]): Tensor[T, TT]
  @native @ByVal def dot[T, TT <: TensorType](@ByRef self: Tensor[T, TT], @ByRef other: Tensor[T, TT]): Tensor[T, TT]
  @native @ByVal def add[T, TT <: TensorType](@ByRef self: Tensor[T, TT], @ByRef other: Tensor[T, TT]): Tensor[T, TT]
  @native @ByVal def max[T, TT <: TensorType](@ByRef self: Tensor[T, TT], @ByRef other: Tensor[T, TT]): Tensor[T, TT]

  @native @ByVal def threshold[T, TT <: TensorType](@ByRef self: Tensor[T, TT], @ByVal threshold: Scalar[T], @ByVal value: Scalar[T]): Tensor[T, TT]

  implicit class MathTensor[T: ClassTag, TT <: TensorType](self: Tensor[T, TT]) {
    def cos(): Tensor[T, TT] = new Tensor(Math.cos(self))
    def sin(): Tensor[T, TT] = new Tensor(Math.sin(self))
    def exp(): Tensor[T, TT] = new Tensor(Math.exp(self))
    def tanh(): Tensor[T, TT] = new Tensor(Math.tanh(self))
    def sigmoid(): Tensor[T, TT] = new Tensor(Math.sigmoid(self))
    def log(): Tensor[T, TT] = new Tensor(Math.log(self))


    def mean(): Tensor[T, TT] = new Tensor(Math.mean(self))
    def mean(dim: Array[Int], keepdim: Boolean = false): Tensor[T, TT] = new Tensor(Math.mean(self, IntList(dim), keepdim))
    def maximum(dim: Long, keepdim: Boolean = false): Tensor[T, TT] = new Tensor(Math.maximum(self, dim, keepdim))
    def argmax(dim: Long, keepdim: Boolean = false): Tensor[Long, TT] = new Tensor(Math.argmax(self, dim, keepdim))
    def softmax(dim: Long): Tensor[T, TT] = new Tensor(Math.softmax(self, dim))

    def matmul(other: Tensor[T, TT]) = new Tensor[T, TT]( Math.matmul(self, other) )
    def dot(other: Tensor[T, TT]) = new Tensor[T, TT]( Math.dot(self, other) )
    def add(other: Tensor[T, TT]) = new Tensor[T, TT]( Math.add(self, other) )
    def maximum(other: Tensor[T, TT]) = new Tensor[T, TT]( Math.max(self, other) )

    def threshold(threshold: Scalar[T], value: Scalar[T]): Tensor[T, TT] = new Tensor( Math.threshold(self, threshold, value) )
  }

  implicit class MathScalar[T: ClassTag](self: Scalar[T]) {
    def +[TT <: TensorType](t: Tensor[T, TT]): Tensor[T, TT] = t + self
    def -[TT <: TensorType](t: Tensor[T, TT]): Tensor[T, TT] = -t + self
    def *[TT <: TensorType](t: Tensor[T, TT]): Tensor[T, TT] = t * self
    def /[TT <: TensorType](t: Tensor[T, TT])(implicit num: Numeric[T]): Tensor[T, TT] = t.**(num.fromInt(-1)) * self

    def +(t: Scalar[T])(implicit num: Numeric[T]): Scalar[T] = new Scalar[T](num.plus(self.getValue, t.getValue))
    def -(t: Scalar[T])(implicit num: Numeric[T]): Scalar[T] = new Scalar[T](num.minus(self.getValue, t.getValue))

  }


  implicit class MathT[T: ClassTag](self: T) {
    def +[TT <: TensorType](t: Tensor[T, TT]): Tensor[T, TT] = t + self
    def -[TT <: TensorType](t: Tensor[T, TT]): Tensor[T, TT] = -t + self
    def *[TT <: TensorType](t: Tensor[T, TT]): Tensor[T, TT] = t * self
    def /[TT <: TensorType](t: Tensor[T, TT])(implicit num: Numeric[T]): Tensor[T, TT] = t.**(num.fromInt(-1)) * self

    def +(t: Scalar[T])(implicit num: Numeric[T]): Scalar[T] = new Scalar[T](num.plus(self, t.getValue))
    def -(t: Scalar[T])(implicit num: Numeric[T]): Scalar[T] = new Scalar[T](num.minus(self, t.getValue))

  }



}
