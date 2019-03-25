
package torch_scala.api



import torch_scala.api.aten.TensorType

import scala.annotation.implicitNotFound

/**
  * @author Emmanouil Antonios Platanios
  */
package object types {
  //region Value Classes

  // TODO: [TYPES] Add some useful functionality to the following types.

  case class Half(data: Short) extends AnyVal
  case class TruncatedHalf(data: Short) extends AnyVal
  case class ComplexHalf(real: Half, imaginary: Half)
  case class ComplexFloat(real: Float, imaginary: Float)
  case class ComplexDouble(real: Double, imaginary: Double)
  case class UByte(data: Byte) extends AnyVal
  case class UShort(data: Short) extends AnyVal
  case class UInt(data: Int) extends AnyVal
  case class ULong(data: Long) extends AnyVal


  //endregion Value Classes

  //region Data Type Instances



  val STRING    : DataType[String]        = DataType[String]("String", 11, None)
  val BOOLEAN   : DataType[Boolean]       = DataType[Boolean]("Boolean", 12, Some(1))
  val FLOAT16   : DataType[Half]          = DataType[Half]("Half", 5, Some(2))
  val FLOAT32   : DataType[Float]         = DataType[Float]("Float", 6, Some(4))
  val FLOAT64   : DataType[Double]        = DataType[Double]("Double", 7, Some(8))
  val COMPLEX32 : DataType[ComplexHalf]  = DataType[ComplexHalf]("ComplexHalf", 8, Some(4))
  val COMPLEX64 : DataType[ComplexFloat]  = DataType[ComplexFloat]("ComplexFloat", 9, Some(8))
  val COMPLEX128: DataType[ComplexDouble] = DataType[ComplexDouble]("ComplexDouble", 10, Some(16))
  val INT8      : DataType[Byte]          = DataType[Byte]("Byte", 0, Some(1))
  val CHAR      : DataType[Char]          = DataType[Char]("Char", 1, Some(1))
  val INT16     : DataType[Short]         = DataType[Short]("Short", 2, Some(2))
  val INT32     : DataType[Int]           = DataType[Int]("Int", 3, Some(4))
  val INT64     : DataType[Long]          = DataType[Long]("Long", 4, Some(8))
  val UINT8     : DataType[UByte]         = DataType[UByte]("UByte", 13, Some(1))
  val UINT16    : DataType[UShort]        = DataType[UShort]("UShort", 14, Some(2))
  val UINT32    : DataType[UInt]          = DataType[UInt]("UInt", 15, Some(4))
  val UINT64    : DataType[ULong]         = DataType[ULong]("ULong", 16, Some(8))


  //endregion Data Type Instances

  //region Type Traits

  @implicitNotFound(msg = "Cannot prove that ${T} is a supported Torch data type.")
  trait DT[T] {
    @inline def dataType: DataType[T]
  }

  @implicitNotFound(msg = "Cannot prove that ${TT} is a supported Tensor type.")
  trait DTT[TT <: TensorType] {
    @inline def tensorType: TT
  }


  object DT {
    def apply[T: DT]: DT[T] = {
      implicitly[DT[T]]
    }

    def fromDataType[T](dataType: DataType[T]): DT[T] = {
      val providedDataType = dataType
      new DT[T] {
        override def dataType: DataType[T] = {
          providedDataType
        }
      }
    }

    implicit val stringEvDT : DT[String]  = fromDataType(STRING)
    implicit val booleanEvDT: DT[Boolean] = fromDataType(BOOLEAN)
    implicit val floatEvDT  : DT[Float]   = fromDataType(FLOAT32)
    implicit val intEvDT    : DT[Int]     = fromDataType(INT32)
    implicit val longEvDT   : DT[Long]    = fromDataType(INT64)
    implicit val doubleEvDT: DT[Double] = fromDataType(FLOAT64)
    implicit val byteEvDT  : DT[Byte]   = fromDataType(INT8)
    implicit val shortEvDT : DT[Short]  = fromDataType(INT16)
  }


  //region Union Types Support

  type ![A] = A => Nothing
  type !![A] = ![![A]]

  trait Disjunction[T] {
    type or[S] = Disjunction[T with ![S]]
    type create = ![T]
  }

  type Union[T] = {
    type or[S] = Disjunction[![T]]#or[S]
  }

  type Contains[S, T] = !![S] <:< T

  //endregion Union Types Support

  type FloatOrDouble = Union[Float]#or[Double]#create
  type HalfOrFloat = Union[Half]#or[Float]#create
  type HalfOrFloatOrDouble = Union[Half]#or[Float]#or[Double]#create
  type TruncatedHalfOrFloatOrDouble = Union[TruncatedHalf]#or[Float]#or[Double]#create
  type TruncatedHalfOrHalfOrFloat = Union[TruncatedHalf]#or[Half]#or[Float]#create
  type Decimal = Union[TruncatedHalf]#or[Half]#or[Float]#or[Double]#create
  type IntOrLong = Union[Int]#or[Long]#create
  type IntOrLongOrFloatOrDouble = Union[Int]#or[Long]#or[Float]#or[Double]#create
  type IntOrLongOrHalfOrFloatOrDouble = Union[Int]#or[Long]#or[Half]#or[Float]#or[Double]#create
  type IntOrLongOrUByte = Union[Int]#or[Long]#or[UByte]#create
  type SignedInteger = Union[Byte]#or[Short]#or[Int]#or[Long]#create
  type UnsignedInteger = Union[UByte]#or[UShort]#or[UInt]#or[ULong]#create
  type UByteOrUShort = Union[UByte]#or[UShort]#create
  type Integer = Union[Byte]#or[Short]#or[Int]#or[Long]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#create
  type StringOrInteger = Union[String]#or[Byte]#or[Short]#or[Int]#or[Long]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#create
  type StringOrFloatOrLong = Union[String]#or[Float]#or[Long]#create
  type StringOrIntOrUInt = Union[String]#or[Int]#or[UInt]#create
  type Real = Union[TruncatedHalf]#or[Half]#or[Float]#or[Double]#or[Byte]#or[Short]#or[Int]#or[Long]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#create
  type Complex = Union[ComplexFloat]#or[ComplexDouble]#create
  type NotQuantized = Union[TruncatedHalf]#or[Half]#or[Float]#or[Double]#or[Byte]#or[Short]#or[Int]#or[Long]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[ComplexFloat]#or[ComplexDouble]#create
  type Numeric = Union[TruncatedHalf]#or[Half]#or[Float]#or[Double]#or[Byte]#or[Short]#or[Int]#or[Long]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[ComplexFloat]#or[ComplexDouble]#create
  type BooleanOrNumeric = Union[Boolean]#or[Half]#or[Float]#or[Double]#or[Byte]#or[Short]#or[Int]#or[Long]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[ComplexFloat]#or[ComplexDouble]#create

  type IsFloatOrDouble[T] = Contains[T, FloatOrDouble]
  type IsHalfOrFloat[T] = Contains[T, HalfOrFloat]
  type IsHalfOrFloatOrDouble[T] = Contains[T, HalfOrFloatOrDouble]
  type IsTruncatedHalfOrFloatOrDouble[T] = Contains[T, TruncatedHalfOrFloatOrDouble]
  type IsTruncatedHalfOrHalfOrFloat[T] = Contains[T, TruncatedHalfOrHalfOrFloat]
  type IsDecimal[T] = Contains[T, Decimal]
  type IsIntOrLong[T] = Contains[T, IntOrLong]
  type IsIntOrLongOrFloatOrDouble[T] = Contains[T, IntOrLongOrFloatOrDouble]
  type IsIntOrLongOrHalfOrFloatOrDouble[T] = Contains[T, IntOrLongOrHalfOrFloatOrDouble]
  type IsIntOrLongOrUByte[T] = Contains[T, IntOrLongOrUByte]
  type IsIntOrUInt[T] = Contains[T, Integer]
  type IsUByteOrUShort[T] = Contains[T, UByteOrUShort]
  type IsStringOrInteger[T] = Contains[T, StringOrInteger]
  type IsStringOrFloatOrLong[T] = Contains[T, StringOrFloatOrLong]
  type IsStringOrIntOrUInt[T] = Contains[T, StringOrIntOrUInt]
  type IsReal[T] = Contains[T, Real]
  type IsComplex[T] = Contains[T, Complex]
  type IsNotQuantized[T] = Contains[T, NotQuantized]
  type IsNumeric[T] = Contains[T, Numeric]
  type IsBooleanOrNumeric[T] = Contains[T, BooleanOrNumeric]

  object IsFloatOrDouble {
    def apply[T: IsFloatOrDouble]: IsFloatOrDouble[T] = implicitly[IsFloatOrDouble[T]]
  }

  object IsHalfOrFloat {
    def apply[T: IsHalfOrFloat]: IsHalfOrFloat[T] = implicitly[IsHalfOrFloat[T]]
  }

  object IsHalfOrFloatOrDouble {
    def apply[T: IsHalfOrFloatOrDouble]: IsHalfOrFloatOrDouble[T] = implicitly[IsHalfOrFloatOrDouble[T]]
  }

  object IsTruncatedHalfOrFloatOrDouble {
    def apply[T: IsTruncatedHalfOrFloatOrDouble]: IsTruncatedHalfOrFloatOrDouble[T] = implicitly[IsTruncatedHalfOrFloatOrDouble[T]]
  }

  object IsTruncatedHalfOrHalfOrFloat {
    def apply[T: IsTruncatedHalfOrHalfOrFloat]: IsTruncatedHalfOrHalfOrFloat[T] = implicitly[IsTruncatedHalfOrHalfOrFloat[T]]
  }

  object IsDecimal {
    def apply[T: IsDecimal]: IsDecimal[T] = implicitly[IsDecimal[T]]
  }

  object IsIntOrLong {
    def apply[T: IsIntOrLong]: IsIntOrLong[T] = implicitly[IsIntOrLong[T]]
  }

  object IsIntOrLongOrFloatOrDouble {
    def apply[T: IsIntOrLongOrFloatOrDouble]: IsIntOrLongOrFloatOrDouble[T] = implicitly[IsIntOrLongOrFloatOrDouble[T]]
  }

  object IsIntOrLongOrHalfOrFloatOrDouble {
    def apply[T: IsIntOrLongOrHalfOrFloatOrDouble]: IsIntOrLongOrHalfOrFloatOrDouble[T] = implicitly[IsIntOrLongOrHalfOrFloatOrDouble[T]]
  }

  object IsIntOrLongOrUByte {
    def apply[T: IsIntOrLongOrUByte]: IsIntOrLongOrUByte[T] = implicitly[IsIntOrLongOrUByte[T]]
  }

  object IsIntOrUInt {
    def apply[T: IsIntOrUInt]: IsIntOrUInt[T] = implicitly[IsIntOrUInt[T]]
  }

  object IsUByteOrUShort {
    def apply[T: IsUByteOrUShort]: IsUByteOrUShort[T] = implicitly[IsUByteOrUShort[T]]
  }

  object IsStringOrInteger {
    def apply[T: IsStringOrInteger]: IsStringOrInteger[T] = implicitly[IsStringOrInteger[T]]
  }

  object IsStringOrFloatOrLong {
    def apply[T: IsStringOrFloatOrLong]: IsStringOrFloatOrLong[T] = implicitly[IsStringOrFloatOrLong[T]]
  }

  object IsReal {
    def apply[T: IsReal]: IsReal[T] = implicitly[IsReal[T]]
  }

  object IsComplex {
    def apply[T: IsComplex]: IsComplex[T] = implicitly[IsComplex[T]]
  }

  object IsNotQuantized {
    def apply[T: IsNotQuantized]: IsNotQuantized[T] = implicitly[IsNotQuantized[T]]
  }

  object IsNumeric {
    def apply[T: IsNumeric]: IsNumeric[T] = implicitly[IsNumeric[T]]
  }

  object IsBooleanOrNumeric {
    def apply[T: IsBooleanOrNumeric]: IsBooleanOrNumeric[T] = implicitly[IsBooleanOrNumeric[T]]
  }
}
