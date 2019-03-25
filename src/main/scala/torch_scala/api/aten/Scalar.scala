package torch_scala.api.aten

import javax.activation.UnsupportedDataTypeException
import org.bytedeco.javacpp.{DoublePointer, FloatPointer, Pointer, ShortPointer}
import org.bytedeco.javacpp.annotation._
import torch_scala.NativeLoader
import torch_scala.api.types.Half

import scala.reflect.ClassTag



@Platform(include = Array("torch/all.h"))
@Namespace("at") @NoOffset @Name(Array("Scalar")) class Scalar[T](value: T) extends Pointer with NativeLoader {
  @native def allocate(v: Int): Unit
  @native def allocate(v: Float): Unit
  @native def allocate(@Cast(Array("long")) v: Long): Unit
  @native def allocate(v: Double): Unit

  value match {
    case v: Int => allocate(v)
    case v: Float => allocate(v)
    case v: Long => allocate(v)
    case v: Double => allocate(v)
    case _ => throw new UnsupportedDataTypeException()
  }

  def getValue: T = value

  @native @Name(Array("to<int>")) def toInt(): Int
  @native @Name(Array("to<float>")) def toFloat(): Float
  @native @Cast(Array("long")) @Name(Array("to<long>")) def toLong(): Long
  @native @Name(Array("to<double>")) def toDouble(): Double

}


