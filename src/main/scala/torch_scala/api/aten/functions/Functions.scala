package torch_scala.api.aten.functions

import org.bytedeco.javacpp.annotation._
import org.bytedeco.javacpp.{FunctionPointer, LongPointer, Pointer}
import torch_scala.NativeLoader
import torch_scala.api.aten._
import torch_scala.api.exception.InvalidDataTypeException
import torch_scala.api.types._

@Platform(include = Array("helper.h",
                          "torch/all.h",
                          "<complex>"))
@Namespace("at")
@NoOffset object Functions extends NativeLoader {


  @Opaque class Type() extends Pointer(null.asInstanceOf[Pointer]) {
    allocate()
    @native def allocate(): Unit
  }


  @native def int_list(@Cast(Array("size_t")) size: Int, data: Array[Int]): IntList

  @native @ByVal def arange[T, TT <: TensorType](@ByVal start: Scalar[Long], @ByVal end: Scalar[Long], @ByRef options: TensorOptions[T, TT]): Tensor[T, TT]
  @native @ByVal def arange[T, TT <: TensorType](@ByVal start: Scalar[Long], @ByVal end: Scalar[Long], @ByVal step: Scalar[Long], @Const @ByRef options: TensorOptions[T, TT]): Tensor[T, TT]
  @native @ByVal def arange(@ByVal start: Scalar[Long], @ByVal end: Scalar[Long]): Tensor[Float, CPU]
  @native @ByVal def arange(@ByVal start: Scalar[Long], @ByVal end: Scalar[Long], @ByVal step: Scalar[Long]): Tensor[Float, CPU]

  @native @ByVal def tensor[TT <: TensorType](@ByVal values: ArrayRefInt)(implicit @Const @ByRef options: TensorOptions[Int, TT]): Tensor[Int, TT]
  @native @ByVal def tensor[TT <: TensorType](@ByVal values: ArrayRefFloat)(implicit @Const @ByRef options: TensorOptions[Float, TT]): Tensor[Float, TT]
  @native @ByVal def tensor[TT <: TensorType](@ByVal values: ArrayRefLong)(implicit @Const @ByRef options: TensorOptions[Long, TT]): Tensor[Long, TT]
  @native @ByVal def tensor[TT <: TensorType](@ByVal values: ArrayRefDouble)(implicit @Const @ByRef options: TensorOptions[Double, TT]): Tensor[Double, TT]
  @native @ByVal def tensor[TT <: TensorType](@ByVal values: ArrayRefByte)(implicit @Const @ByRef options: TensorOptions[Byte, TT]): Tensor[Byte, TT]
  def apply[T, TT <: TensorType](data: ArrayRef[T, _])(options: TensorOptions[T, TT]): Tensor[T, TT] = data.dataType match {
    case INT32 => tensor[TT](data.asInstanceOf[ArrayRefInt])(options.asInstanceOf[TensorOptions[Int, TT]]).asInstanceOf[Tensor[T, TT]]
    case FLOAT32 => tensor(data.asInstanceOf[ArrayRefFloat])(options.asInstanceOf[TensorOptions[Float, TT]]).asInstanceOf[Tensor[T, TT]]
    case INT64 => tensor(data.asInstanceOf[ArrayRefLong])(options.asInstanceOf[TensorOptions[Long, TT]]).asInstanceOf[Tensor[T, TT]]
    case FLOAT64 => tensor(data.asInstanceOf[ArrayRefDouble])(options.asInstanceOf[TensorOptions[Double, TT]]).asInstanceOf[Tensor[T, TT]]
    case INT8 => tensor(data.asInstanceOf[ArrayRefByte])(options.asInstanceOf[TensorOptions[Byte, TT]]).asInstanceOf[Tensor[T, TT]]
    case _ => throw InvalidDataTypeException("type:" + data.dataType)
  }

  @native @ByVal def zeros[T, TT <: TensorType](@ByVal size: IntList)(implicit @Const @ByRef options: TensorOptions[T, TT]): Tensor[T, TT]
  @native @ByVal def zeros_like[T, TT <: TensorType](@Const @ByRef self: Tensor[T, TT]): Tensor[T, TT]
  @native @ByVal def zeros_like[T, TT <: TensorType](@Const @ByRef self: Tensor[T, TT], @Const @ByRef options: TensorOptions[T, TT]): Tensor[T, TT]

  @native @ByVal def ones[T, TT <: TensorType](@ByVal size: IntList)(implicit @Const @ByRef options: TensorOptions[T, TT]): Tensor[T, TT]
  @native @ByVal def ones_like[T, TT <: TensorType](@Const @ByRef self: Tensor[T, TT]): Tensor[T, TT]
  @native @ByVal def ones_like[T, TT <: TensorType](@Const @ByRef self: Tensor[T, TT], @Const @ByRef options: TensorOptions[T, TT]): Tensor[T, TT]

  @native @ByVal def randn[T, TT <: TensorType](@ByVal size: IntList)(implicit @Const @ByRef options: TensorOptions[T, TT]): Tensor[T, TT]
  @native @ByVal def randn_like[T, TT <: TensorType](@Const @ByRef self: Tensor[T, TT]): Tensor[T, TT]
  @native @ByVal def randn_like[T, TT <: TensorType](@Const @ByRef self: Tensor[T, TT], @Const @ByRef options: TensorOptions[T, TT]): Tensor[T, TT]

  @native @ByVal def full[T, TT <: TensorType](@ByVal size: IntList, @ByVal fill_value: Scalar[T])(implicit @Const @ByRef options: TensorOptions[T, TT]): Tensor[T, TT]

  @native @ByVal def randint[TT <: TensorType](@Cast(Array("long")) low: Long,
                                                  @Cast(Array("long")) high: Long,
                                                  @ByVal size: IntList)(implicit @Const @ByRef options: TensorOptions[Long, TT]): Tensor[Long, TT]


  class Deallocator_Pointer(p: Pointer) extends FunctionPointer(p) {
    @Name(Array("deleter")) def call(data: Pointer): Unit = {
      Pointer.free(data)
      println("delete tensor")
    }
  }

  @native @ByVal def from_blob[T](
    data: Pointer,
    @ByVal sizes: IntList,
    @Cast(Array("const std::function<void(void*)>")) deleter: Deallocator_Pointer)(implicit @Const @ByRef options: TensorOptions[T, CPU]): Tensor[T, CPU]

  @native @ByVal def from_blob[T](
                                   data: Pointer,
                                   @ByVal sizes: IntList,
                                   @ByVal strides: IntList,
                                   deleter: Deallocator_Pointer)(implicit @Const @ByRef options: TensorOptions[T, CPU]): Tensor[T, CPU]

}
