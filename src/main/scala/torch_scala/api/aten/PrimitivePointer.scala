package torch_scala.api.aten

import org.bytedeco.javacpp._

import scala.reflect.ClassTag


object PrimitivePointer {
  implicit class PrimitivePointer[PT <: Pointer](data: PT) {
    def asArray[T: ClassTag](num_elements: Int): Array[T] = data match {
      case dd: IntPointer => Array.range(0, num_elements).map(i => dd.get(i).asInstanceOf[T]).toArray
      case dd: LongPointer => Array.range(0, num_elements).map(i => dd.get(i).asInstanceOf[T]).toArray
      case dd: FloatPointer => Array.range(0, num_elements).map(i => dd.get(i).asInstanceOf[T]).toArray
      case dd: DoublePointer => Array.range(0, num_elements).map(i => dd.get(i).asInstanceOf[T]).toArray
      case dd: ShortPointer => Array.range(0, num_elements).map(i => dd.get(i).asInstanceOf[T]).toArray
      case dd: BytePointer => Array.range(0, num_elements).map(i => dd.get(i).asInstanceOf[T]).toArray
    }
  }

  def apply[T](data: Array[T]): Pointer = data.head match {
    case h: Float => new FloatPointer(data.asInstanceOf[Array[Float]]:_*).asInstanceOf[Pointer]
    case h: Int => new IntPointer(data.asInstanceOf[Array[Int]]:_*).asInstanceOf[Pointer]
    case h: Long => new LongPointer(data.asInstanceOf[Array[Long]]:_*).asInstanceOf[Pointer]
    case h: Double => new DoublePointer(data.asInstanceOf[Array[Double]]:_*).asInstanceOf[Pointer]
    case h: Byte => new BytePointer(data.asInstanceOf[Array[Byte]]:_*).asInstanceOf[Pointer]
  }
}