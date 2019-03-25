package torch_scala.api.types



import java.nio.ByteBuffer
import java.nio.charset.StandardCharsets

/** Represents the data type of the elements in a tensor.
  *
  * @param  name      Name of this data type (mainly useful for logging purposes).
  * @param  cValue    Represents this data type in the `TF_DataType` enum of the TensorFlow C API.
  * @param  byteSize  Size in bytes of each value with this data type. Set to `None` if the size is not fixed.
  * @param  protoType ProtoBuf data type used in serialized representations of tensors.
  * @tparam T         Corresponding Scala type for this TensorFlow data type.
  *
  * @author Emmanouil Antonios Platanios
  */
case class DataType[+T] private[types](
                                       name: String,
                                       private[api] val cValue: Int,
                                       byteSize: Option[Int]
                                     ) {
  //region Data Type Properties

  //endregion Data Type Properties

  //region Data Type Set Helper Methods

  /** Returns `true` if this data type represents a non-quantized floating-point data type. */
  def isFloatingPoint: Boolean = {
    Set[DataType[Any]](FLOAT16, FLOAT32, FLOAT64)
      .contains(this.asInstanceOf[DataType[Any]])
  }

  /** Returns `true` if this data type represents a complex data types. */
  def isComplex: Boolean = {
    Set[DataType[Any]](COMPLEX64, COMPLEX128)
      .contains(this.asInstanceOf[DataType[Any]])
  }

  /** Returns `true` if this data type represents a non-quantized integer data type. */
  def isInteger: Boolean = {
    Set[DataType[Any]](INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64)
      .contains(this.asInstanceOf[DataType[Any]])
  }


  /** Returns `true` if this data type represents a non-quantized unsigned data type. */
  def isUnsigned: Boolean = {
    Set[DataType[Any]](UINT8, UINT16, UINT32, UINT64)
      .contains(this.asInstanceOf[DataType[Any]])
  }

  /** Returns `true` if this data type represents a numeric data type. */
  def isNumeric: Boolean = {
    isFloatingPoint || isComplex || isInteger || isUnsigned
  }

  /** Returns `true` if this data type represents a boolean data type. */
  def isBoolean: Boolean = {
    this == BOOLEAN
  }

  //endregion Data Type Set Helper Methods

  override def toString: String = {
    name
  }

  override def equals(that: Any): Boolean = that match {
    case that: DataType[T] => this.cValue == that.cValue
    case _ => false
  }

  override def hashCode: Int = {
    cValue
  }

  def zero: T = {
    val res = cValue match {
      case FLOAT32.cValue => 0f
      case FLOAT64.cValue => 0d
      case INT8.cValue => 0.toByte
      case INT16.cValue => 0.toShort
      case INT32.cValue => 0
      case INT64.cValue => 0l
      case value => throw new IllegalArgumentException(
        s"Data type C value '$value' is not valid for zero() function.")
    }
    res.asInstanceOf[T]
  }

  def one: T = {
    val res = cValue match {
      case FLOAT32.cValue => 1f
      case FLOAT64.cValue => 1d
      case INT8.cValue => 1.toByte
      case INT16.cValue => 1.toShort
      case INT32.cValue => 1
      case INT64.cValue => 1l
      case value => throw new IllegalArgumentException(
        s"Data type C value '$value' is not valid for ones() function.")
    }
    res.asInstanceOf[T]
  }


  def cast(v: Int): T = {
    val res = cValue match {
      case FLOAT32.cValue => v.toFloat
      case FLOAT64.cValue => v.toDouble
      case INT32.cValue => v
      case INT64.cValue => v.toLong
      case value => throw new IllegalArgumentException(
        s"Data type C value '$value' is not valid for cast() function.")
    }
    res.asInstanceOf[T]
  }

  def cast(v: Double): T = {
    val res = cValue match {
      case FLOAT32.cValue => v.toFloat
      case FLOAT64.cValue => v
      case value => throw new IllegalArgumentException(
        s"Data type C value '$value' is not valid for cast() function.")
    }
    res.asInstanceOf[T]
  }


}

/** Contains all supported data types along with some helper functions for dealing with them. */
object DataType {
  //region Helper Methods

  /** Returns the data type that corresponds to the provided C value.
    *
    * By C value here we refer to an integer representing a data type in the `TF_DataType` enum of the TensorFlow C
    * API.
    *
    * @param  cValue C value.
    * @return Data type corresponding to the provided C value.
    * @throws IllegalArgumentException If an invalid C value is provided.
    */
  @throws[IllegalArgumentException]
  private[api] def fromCValue[T](cValue: Int): DataType[T] = {
    val dataType = cValue match {
      case BOOLEAN.cValue => BOOLEAN
      case STRING.cValue => STRING
      case FLOAT16.cValue => FLOAT16
      case FLOAT32.cValue => FLOAT32
      case FLOAT64.cValue => FLOAT64
      case COMPLEX64.cValue => COMPLEX64
      case COMPLEX128.cValue => COMPLEX128
      case INT8.cValue => INT8
      case INT16.cValue => INT16
      case INT32.cValue => INT32
      case INT64.cValue => INT64
      case UINT8.cValue => UINT8
      case UINT16.cValue => UINT16
      case UINT32.cValue => UINT32
      case UINT64.cValue => UINT64
      case value => throw new IllegalArgumentException(
        s"Data type C value '$value' is not recognized in Scala.")
    }
    dataType.asInstanceOf[DataType[T]]
  }

  //endregion Helper Methods
}
