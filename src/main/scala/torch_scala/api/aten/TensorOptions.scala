package torch_scala.api.aten

import org.bytedeco.javacpp.{FloatPointer, Pointer}
import org.bytedeco.javacpp.annotation._
import torch_scala.NativeLoader
import torch_scala.api.types._

import scala.reflect.runtime.universe.TypeTag

@Platform(include = Array("torch/all.h"))
@Namespace("at") class TensorOptions[T, TT <: TensorType]() extends Pointer(null.asInstanceOf[Pointer]) with NativeLoader {
  allocate()
  @native def allocate(): Unit
  @native @ByVal def device(@ByRef d: Device[TT]): TensorOptions[T, TT]
  @native @ByVal def device_index(@Cast(Array("int16_t")) device_index: Short): TensorOptions[T, TT]
}


@Platform(include = Array("helper.h"))
@Namespace("at")
@NoOffset object TensorOptions extends NativeLoader {

  def apply[T, TT <: TensorType](device: Device[TT])(implicit dtype: DT[T]): TensorOptions[T, TT] = {
    create_options[T, TT](dtype.dataType).device(device)
  }


  @native @ByVal private def create_options[T, TT <: TensorType](dtype: Int): TensorOptions[T, TT]
  def create_options[T, TT <: TensorType](dtype: DataType[T]): TensorOptions[T, TT] = create_options(dtype.cValue)

//  {0, kByte},
//  {1, kChar},
//  {2, kShort},
//  {3, kInt},
//  {4, kLong},
//  {5, kHalf},
//  {6, kFloat},
//  {7, kDouble},
//  {8, kComplexHalf},
//  {9, kComplexFloat},
//  {10, kComplexDouble}

  implicit val byteTensorOptions: TensorOptions[Byte, CPU] = create_options(INT8)
  implicit val charTensorOptions: TensorOptions[Char, CPU] = create_options(CHAR)
  implicit val shortTensorOptions: TensorOptions[Short, CPU] = create_options(INT16)
  implicit val intTensorOptions: TensorOptions[Int, CPU] = create_options(INT32)
  implicit val longTensorOptions: TensorOptions[Long, CPU] = create_options(INT64)
  implicit val halfTensorOptions: TensorOptions[Half, CPU] = create_options(FLOAT16)
  implicit val floatTensorOptions: TensorOptions[Float, CPU] = create_options(FLOAT32)
  implicit val doubleTensorOptions: TensorOptions[Double, CPU] = create_options(FLOAT64)
  implicit val complexHalfTensorOptions: TensorOptions[ComplexHalf, CPU] = create_options(COMPLEX32)
  implicit val complexFloatTensorOptions: TensorOptions[ComplexFloat, CPU] = create_options(COMPLEX64)
  implicit val complexDoubleTensorOptions: TensorOptions[ComplexDouble, CPU] = create_options(COMPLEX128)

  implicit val cudabyteTensorOptions: TensorOptions[Byte, CUDA] = create_options(INT8).device(CudaDevice)
  implicit val cudacharTensorOptions: TensorOptions[Char, CUDA] = create_options(CHAR).device(CudaDevice)
  implicit val cudashortTensorOptions: TensorOptions[Short, CUDA] = create_options(INT16).device(CudaDevice)
  implicit val cudaintTensorOptions: TensorOptions[Int, CUDA] = create_options(INT32).device(CudaDevice)
  implicit val cudalongTensorOptions: TensorOptions[Long, CUDA] = create_options(INT64).device(CudaDevice)
  implicit val cudahalfTensorOptions: TensorOptions[Half, CUDA] = create_options(FLOAT16).device(CudaDevice)
  implicit val cudafloatTensorOptions: TensorOptions[Float, CUDA] = create_options(FLOAT32).device(CudaDevice)
  implicit val cudadoubleTensorOptions: TensorOptions[Double, CUDA] = create_options(FLOAT64).device(CudaDevice)
  implicit val cudacomplexHalfTensorOptions: TensorOptions[ComplexHalf, CUDA] = create_options(COMPLEX32).device(CudaDevice)
  implicit val cudacomplexFloatTensorOptions: TensorOptions[ComplexFloat, CUDA] = create_options(COMPLEX64).device(CudaDevice)
  implicit val cudacomplexDoubleTensorOptions: TensorOptions[ComplexDouble, CUDA] = create_options(COMPLEX128).device(CudaDevice)

}