package torch_scala.api.aten

import org.bytedeco.javacpp.{Pointer, _}
import org.bytedeco.javacpp.annotation._
import torch_scala.{NativeLoader, Torch}
import torch_scala.api.aten
import torch_scala.api.aten.functions.Functions.Deallocator_Pointer

import scala.reflect.ClassTag
import torch_scala.api._
import torch_scala.api.exception.InvalidDeviceException
import PrimitivePointer._
import torch_scala.api.aten.functions.Functions
import torch_scala.api.types.{DT, DataType}
import torch_scala.exceptions.UnavailableException

import scala.reflect


@Platform(include = Array("ATen/ATen.h"))
@Namespace("at") @NoOffset class Tensor[T :ClassTag, TT <: TensorType](nt: Tensor[T, TT]) extends Pointer with NativeLoader {
  allocate(nt)

  def tensorType: TT = is_cuda() match {
    case true => new CUDA(if(device().has_index()) device().index() else 0, scalar_type()).asInstanceOf[TT]
    case false => new CPU(scalar_type()).asInstanceOf[TT]
  }

  /** Data type of this tensor. */
  def dataType: DataType[T] = {
    DataType.fromCValue[T](scalar_type())
  }

  private var shape_obj: Shape = Shape.apply(sizesIterator.toArray)

  def shape: Shape = Shape.apply(sizesIterator.toArray)

  @native private def allocate(@ByRef other: Tensor[T, TT]): Unit

  @native @ByVal def options(): TensorOptions[T, TT]
  def long_options(): TensorOptions[Long, TT] = TensorOptions.apply[Long, TT](device())

  @native @Cast(Array("long")) def dim: Long

  @native @Cast(Array("const char *")) override def toString: String

  @native @Cast(Array("long")) def storage_offset: Long

  @native def defined: Boolean

  @native def reset(): Unit

  @native def is_same(@ByRef tensor: Tensor[T, TT]): Boolean

  @native @Cast(Array("size_t")) def use_count: Long

  @native @Cast(Array("size_t")) def weak_use_count: Long

  @native def print(): Unit

  @native @ByVal private def sizes: IntList
  private def sizesIterator: Iterator[Int] = {
    val ss = sizes.data()
    (0 until dim.toInt).map(ss.get(_).toInt).toIterator
  }

  def num_elements: Long = {
    val ss = sizes.data()
    (0 until dim.toInt).map(ss.get(_)).product
  }

  @native @ByVal def strides: IntList

  @native @ByVal private def reshape(@ByVal shape: IntList): Tensor[T, TT]
  def reshape(shape: Shape): Tensor[T, TT] = {
    val t = reshape(new IntList(shape.asArray.map(_.toLong)))
    new Tensor(t)
  }

  @native @Cast(Array("long")) def ndimension: Long

  @native def is_contiguous: Boolean

  @native @ByRef def `type`: Functions.Type

  @native @Cast(Array("int8_t")) def scalar_type(): Short


  @native @Name(Array("cpu")) @ByVal private def to_cpu(): Tensor[T, CPU]
  @native @Name(Array("cuda")) @ByVal private def to_cuda(): Tensor[T, CUDA]
  def cpu(): Tensor[T, CPU] = new Tensor(to_cpu())
  def cuda(): Tensor[T, CUDA] = {
    if(!Torch.is_available()) {
      InvalidDeviceException(s"cuda is not available")
    }
    new Tensor(to_cuda())
  }
  def cuda(index: Short): Tensor[T, CUDA] = {
    if(index >= Torch.device_count()) {
      InvalidDeviceException(s"cuda index $index >= available devise count")
    }
    new Tensor( to[T, CUDA](CudaDevice(index), scalar_type()) )
  }

  @native @Name(Array("data<int>")) private def data_int(): IntPointer
  @native @Name(Array("data<float>")) private def data_float(): FloatPointer
  @native @Cast(Array("long long int*")) @Name(Array("data<int64_t>")) private def data_long(): LongPointer
  @native @Name(Array("data<double>")) private def data_double(): DoublePointer
  @native @Cast(Array("int8_t*")) @Name(Array("data<uint8_t>")) private def data_byte(): BytePointer
  def data(): Array[T] = {

    if (is_cuda()) throw new UnavailableException("gpu data is not accessible, use .cpu() method first")

    scalar_type() match {
      case 0 =>
        val dd = data_byte()
        Array.range(0, num_elements.toInt).map(i => dd.get(i.toLong)).asInstanceOf[Array[T]]
      case 3 =>
        val dd = data_int()
        Array.range(0, num_elements.toInt).map(i => dd.get(i.toLong)).asInstanceOf[Array[T]]
      case 6 =>
        val dd = data_float()
        Array.range(0, num_elements.toInt).map(i => dd.get(i.toLong)).asInstanceOf[Array[T]]
      case 4 =>
        val dd = data_long()
        Array.range(0, num_elements.toInt).map(i => dd.get(i.toLong)).asInstanceOf[Array[T]]
      case 7 =>
        val dd = data_double()
        Array.range(0, num_elements.toInt).map(i => dd.get(i.toLong)).asInstanceOf[Array[T]]
    }
  }

  def data_with_shape(): Array[_ >: T with Array[Any]] = {
     if(shape.rank == 1) {
       data()
     } else if (shape.rank > 1) {
       Array.range(0, shape(0)).map(i => get(i).data_with_shape())
     } else {
       Array(item())
     }
  }

  @native @Name(Array("item<float>")) private def item_float(): Float
  @native @Name(Array("item<int>")) private def item_int(): Int
  @native @Name(Array("item<long>")) private def item_long(): Long
  @native @Name(Array("item<double>")) private def item_double(): Double

  def item(): T = {

    if (is_cuda()) throw new UnavailableException("gpu data is not accessible, use .cpu() method first")

    scalar_type() match {
      case 3 => item_int().asInstanceOf[T]
      case 4 => item_long().asInstanceOf[T]
      case 6 => item_float().asInstanceOf[T]
      case 7 => item_double().asInstanceOf[T]
    }
  }

  def scalar(): Scalar[T] = new Scalar[T](item())

  @native @ByVal @Name(Array("operator[]")) private def apply(@ByVal index: Tensor[Long, TT]): Tensor[T, TT]
  @native @ByVal @Name(Array("operator[]")) def get(@Cast(Array("long")) index: Long): Tensor[T, TT]
  @native @ByVal private def index_select(@Cast(Array("long")) dim: Long,  @ByRef index: Tensor[Long, TT]): Tensor[T, TT]

  def select_in_dim(dim: Long, index: Array[Long]): Tensor[T, TT] = {
    implicit val opt: TensorOptions[Long, TT] = TensorOptions.apply[Long, TT](device())
    new Tensor[T, TT](index_select(
      dim,
      Tensor.apply[Long, TT](index)
    ))
  }

  def select(index: Array[Long], otherIndexes: Array[Long]*): Tensor[T, TT] = {
    val selectors: Array[Array[Long]] = Array(index) ++ otherIndexes
    var res = this

    for (i <- selectors.indices) {
      res = res.select_in_dim(i, selectors(i))
    }

    new Tensor[T, TT](res)
  }

  @native def is_cuda(): Boolean

  @native @ByVal def device(): Device[TT]

  @native @ByVal private def slice(@Cast(Array("long")) dim: Long, @Cast(Array("long")) start: Long = 0, @Cast(Array("long")) end: Long = 9223372036854775807l, @Cast(Array("long")) step: Long = 1): Tensor[T, TT]

  def apply(
             firstIndexer: Indexer,
             otherIndexers: Indexer*
           ): Tensor[T, TT] = {
    val stridedSlice = Indexer.toStridedSlice(firstIndexer, otherIndexers: _*)
    val beginTensor: Array[Int] = stridedSlice._1
    val endTensor: Array[Int] = stridedSlice._2
    val stridesTensor: Array[Int] = stridedSlice._3

    val beginMask: Long = stridedSlice._4
    var endMask: Long = stridedSlice._5
    var ellipsisMask: Long = stridedSlice._6
    var newAxisMask: Long = stridedSlice._7
    var shrinkAxisMask: Long = stridedSlice._8

    var res = this

    for (i <- beginTensor.indices) {
      val e = if(endTensor(i) == -1) shape.apply(i) else endTensor(i)
      res = res.slice(i, beginTensor(i), e, stridesTensor(i))
    }

    new Tensor[T, TT](res)
  }


  @native @ByRef private def put_(@ByRef indices: Tensor[Long, TT], @ByRef values: Tensor[T, TT], accumulate: Boolean): Tensor[T, TT]
  def put(indices: Tensor[Long, TT], values: Tensor[T, TT], accumulate: Boolean = false): Tensor[T, TT] = new Tensor(put_(indices, values, accumulate))
  def put(index: Array[Long], value: Array[T]): Tensor[T, TT] = put(Tensor.apply(index, long_options()), Tensor.apply(value, options()), false)
  def put(index: Long, value: Scalar[T]): Tensor[T, TT] = put(Array(index), Array[T](value.getValue))

  @native @ByVal def take(@ByRef indices: Tensor[Long, TT]): Tensor[T, TT]

  @native @Name(Array("operator=")) @ByRef def set(@ByRef other: Tensor[T, TT]): Tensor[T, TT]

  @native @ByVal def ne(@ByRef other: Tensor[T, TT]): Tensor[Byte, TT]
  @native @ByVal def eq(@ByRef other: Tensor[T, TT]): Tensor[Byte, TT]
  @native @ByVal def ge(@ByRef other: Tensor[T, TT]): Tensor[Byte, TT]
  @native @ByVal def le(@ByRef other: Tensor[T, TT]): Tensor[Byte, TT]
  @native @ByVal def gt(@ByRef other: Tensor[T, TT]): Tensor[Byte, TT]
  @native @ByVal def lt(@ByRef other: Tensor[T, TT]): Tensor[Byte, TT]
  @native @ByVal def ne(@ByVal other: Scalar[T]): Tensor[Byte, TT]
  @native @ByVal def eq(@ByVal other: Scalar[T]): Tensor[Byte, TT]
  @native @ByVal def ge(@ByVal other: Scalar[T]): Tensor[Byte, TT]
  @native @ByVal def le(@ByVal other: Scalar[T]): Tensor[Byte, TT]
  @native @ByVal def gt(@ByVal other: Scalar[T]): Tensor[Byte, TT]
  @native @ByVal def lt(@ByVal other: Scalar[T]): Tensor[Byte, TT]

  @native @ByVal def to[T1, TT1 <: TensorType](@Const @ByRef options: TensorOptions[T, TT1], non_blocking: Boolean, copy: Boolean): Tensor[T1, TT1]
  @native @ByVal def to[T1, TT1 <: TensorType](@ByRef other: Tensor[T1, TT1]): Tensor[T1, TT1]
  @native @ByVal private def to[T1, TT1 <: TensorType](@ByVal d: Device[TT1], @Cast(Array("int8_t")) dtype: Short): Tensor[T1, TT1]
  @native @ByVal private def toType[T1](@Cast(Array("c10::ScalarType")) dtype: Short): Tensor[T1, TT]
  def cast[T1](dataType: DataType[T1]): Tensor[T1, TT] = toType[T1](dataType.cValue.toShort)
  def cast[T1](implicit dt: DT[T1]): Tensor[T1, TT] = cast[T1](dt.dataType)

  @native @Name(Array("operator+=")) @ByRef private def addeq(@ByRef other: Tensor[T, TT]): Tensor[T, TT]
  def += (other: Tensor[T, TT]): Tensor[T, TT] = new Tensor(addeq(other))
  @native @Name(Array("operator+=")) @ByRef private def addeq(@ByVal other: Scalar[T]): Tensor[T, TT]
  def += (other: T): Tensor[T, TT] = new Tensor(addeq(other))
  @native @Name(Array("operator-")) @ByVal private def minus(): Tensor[T, TT]
  def - (): Tensor[T, TT] = new Tensor(minus())
  def unary_- : Tensor[T, TT] = new Tensor(minus())
  @native @Name(Array("operator-=")) @ByRef private def subeq(@ByRef other: Tensor[T, TT]): Tensor[T, TT]
  def -= (other: Tensor[T, TT]): Tensor[T, TT] = new Tensor(subeq(other))
  @native @Name(Array("operator-=")) @ByRef private def subeq(@ByVal other: Scalar[T]): Tensor[T, TT]
  def -= (other: T): Tensor[T, TT] = new Tensor(subeq(other))
  @native @Name(Array("operator*=")) @ByRef private def muleq(@ByRef other: Tensor[T, TT]): Tensor[T, TT]
  def *= (other: Tensor[T, TT]): Tensor[T, TT] = new Tensor(muleq(other))
  @native @Name(Array("operator*=")) @ByRef private def muleq(@ByVal other: Scalar[T]): Tensor[T, TT]
  def *= (other: T): Tensor[T, TT] = new Tensor(muleq(other))
  @native @Name(Array("operator/=")) @ByRef private def releq(@ByRef other: Tensor[T, TT]): Tensor[T, TT]
  def /= (other: Tensor[T, TT]): Tensor[T, TT] = new Tensor(releq(other))
  @native @Name(Array("operator/=")) @ByRef private def releq(@ByVal other: Scalar[T]): Tensor[T, TT]
  def /= (other: T): Tensor[T, TT] = new Tensor(releq(other))

  @native @Name(Array("operator+")) @ByVal private def add(@ByRef other: Tensor[T, TT]): Tensor[T, TT]
  def + (other: Tensor[T, TT]): Tensor[T, TT] = new Tensor(add(other))
  @native @Name(Array("operator+")) @ByVal private def add(@ByVal other: Scalar[T]): Tensor[T, TT]
  def + (other: Scalar[T]): Tensor[T, TT] = new Tensor(add(other))
  @native @Name(Array("operator-")) @ByVal private def sub(@ByRef other: Tensor[T, TT]): Tensor[T, TT]
  def - (other: Tensor[T, TT]): Tensor[T, TT] = new Tensor(sub(other))
  @native @Name(Array("operator-")) @ByVal private def sub(@ByVal other: Scalar[T]): Tensor[T, TT]
  def - (other: Scalar[T]): Tensor[T, TT] = new Tensor(sub(other))
  @native @Name(Array("operator*")) @ByVal private def mul(@ByRef other: Tensor[T, TT]): Tensor[T, TT]
  def * (other: Tensor[T, TT]): Tensor[T, TT] = new Tensor(mul(other))
  @native @Name(Array("operator*")) @ByVal private def mul(@ByVal other: Scalar[T]): Tensor[T, TT]
  def * (other: Scalar[T]): Tensor[T, TT] = new Tensor(mul(other))
  @native @Name(Array("operator/")) @ByVal private def rel(@ByRef other: Tensor[T, TT]): Tensor[T, TT]
  def / (other: Tensor[T, TT]): Tensor[T, TT] = new Tensor(rel(other))
  @native @Name(Array("operator/")) @ByVal private def rel(@ByVal other: Scalar[T]): Tensor[T, TT]
  def / (other: Scalar[T]): Tensor[T, TT] = new Tensor(rel(other))

  @native @ByVal private def sum(@ByVal dims: IntList, keepdim: Boolean): Tensor[T, TT]
  @native @ByVal def sum(): Tensor[T, TT]
  def sum(dims: Array[Int], keepdim: Boolean = false): Tensor[T, TT] = {
    val t = sum(new IntList(dims.map(_.toLong)), keepdim)
    new Tensor(t)
  }

  def sum(dims: Int*): Tensor[T, TT] = sum(dims.toArray, false)

  @native @ByVal private def pow(@ByRef other: Tensor[T, TT]): Tensor[T, TT]
  def ** (other: Tensor[T, TT]): Tensor[T, TT] = new Tensor(pow(other))
  @native @ByVal private def pow(@ByVal other: Scalar[T]): Tensor[T, TT]
  def ** (other: Scalar[T]): Tensor[T, TT] = new Tensor(pow(other))


  @native @Name(Array("sqrt")) @ByVal private def sqrt_op(): Tensor[T, TT]
  def sqrt(): Tensor[T, TT] = {
    new Tensor(sqrt_op())
  }

  @native @Name(Array("abs")) @ByVal private def abs_op(): Tensor[T, TT]
  def abs(): Tensor[T, TT] = {
    new Tensor(abs_op())
  }

  @native @ByVal private def t(): Tensor[T, TT]
  def T: Tensor[T, TT] = {
    new Tensor(t())
  }

  @native @Name(Array("max")) @ByVal private def max_op(): Tensor[T, TT]
  def max(): Tensor[T, TT] = {
    new Tensor(max_op())
  }

  @native @Name(Array("min")) @ByVal private def min_op(): Tensor[T, TT]
  def min(): Tensor[T, TT] = {
    new Tensor(min_op())
  }





}


object Tensor {

  def summarize[T, TT <: TensorType](tensor: Tensor[T, TT], maxEntries: Int = 6): String = {
    def summarize[T, TT <: TensorType](tensor: Tensor[T, TT], maxEntries: Int, level: Int): String = tensor.dim match {
      case 0 => tensor.item().toString
      case 1 =>
        val n = tensor.num_elements.toInt
        val slice =
          if (tensor.num_elements <= math.max(maxEntries, 6))
            tensor.data()
          else
            (tensor(0 :: maxEntries / 2).data() :+ "...") ++ tensor(n - maxEntries / 2 :: n).data()
        slice.mkString("[", "\t", "]")
      case _ =>
        val innerSummary = {
          def summarizeSlice(index: Int) = {
            summarize(tensor(index).reshape(tensor.shape(1 ::)), maxEntries, level + 1)
          }

          if (tensor.shape(0) <= math.max(maxEntries, 6))
            for (i <- 0 until tensor.shape(0)) yield summarizeSlice(i)
          else {
            val start = for (i <- 0 until maxEntries / 2) yield summarizeSlice(i)
            val end = for (i <- tensor.shape(0) - maxEntries / 2 until tensor.shape(0)) yield summarizeSlice(i)
            (start :+ "...") ++ end
          }
        }
        val padding = " " * (level + 1)
        val extraLine = if (tensor.dim >= 3) "\n" else ""
        innerSummary.mkString("[", "\n" + extraLine + padding, "]")
    }

    tensor.toString + tensor.shape.toString + "\n"  + summarize(if (tensor.is_cuda()) tensor.cpu() else tensor, maxEntries, 0) + "\n"
  }

  def cpu[T:ClassTag](data: Array[T], shape: Shape)(implicit options: TensorOptions[T, CPU]): Tensor[T, CPU] = {
    val pt = PrimitivePointer(data)
    val nt = Functions.from_blob[T](pt, new IntList(shape.asArray.map(_.toLong)), new Deallocator_Pointer(null))

    new Tensor[T, CPU](nt)

  }

  def apply[T: ClassTag, TT <: TensorType](data: Array[T], options: TensorOptions[T, TT]): Tensor[T, TT] = {

    val nt = Functions.apply[T, TT](ArrayRef(data))(options)
    new Tensor[T, TT](nt.asInstanceOf[Tensor[T, TT]])

  }

  def apply[T: ClassTag, TT <: TensorType](data: Array[T])(implicit options: TensorOptions[T, TT]): Tensor[T, TT] = {
    apply(data, options)
  }

  def apply[T: ClassTag, TT <: TensorType](data: T*)(implicit options: TensorOptions[T, TT]): Tensor[T, TT] = {
    apply(data.toArray)
  }

  def cpu[T: ClassTag](data: Array[T])(implicit options: TensorOptions[T, CPU]): Tensor[T, CPU] = apply[T, CPU](data)

  def cuda[T: ClassTag](data: Array[T])(implicit options: TensorOptions[T, CUDA]): Tensor[T, CUDA] = apply[T, CUDA](data)

  def ones[T: ClassTag, TT <: TensorType](shape: Shape)(implicit options: TensorOptions[T, TT]): Tensor[T, TT] = {
    new Tensor[T, TT](Functions.ones(new IntList(shape.asArray.map(_.toLong))))
  }

  def ones_like[T: ClassTag, TT <: TensorType](self: Tensor[T, TT]): Tensor[T, TT] = {
    new Tensor[T, TT](Functions.ones_like(self))
  }

  def zeros[T: ClassTag, TT <: TensorType](shape: Shape)(implicit options: TensorOptions[T, TT]): Tensor[T, TT] = {
    new Tensor[T, TT](Functions.zeros(new IntList(shape.asArray.map(_.toLong))))
  }

  def zeros_like[T: ClassTag, TT <: TensorType](self: Tensor[T, TT]): Tensor[T, TT] = {
    new Tensor[T, TT](Functions.zeros_like(self))
  }

  def randn[T: ClassTag, TT <: TensorType](shape: Shape)(implicit options: TensorOptions[T, TT]): Tensor[T, TT] = {
    new Tensor[T, TT](Functions.randn(new IntList(shape.asArray.map(_.toLong))))
  }

  def randn_like[T: ClassTag, TT <: TensorType](self: Tensor[T, TT]): Tensor[T, TT] = {
    new Tensor[T, TT](Functions.randn_like(self))
  }

  def arange[T: ClassTag, TT <: TensorType](from: Scalar[Long], to: Scalar[Long])(implicit options: TensorOptions[T, TT]): Tensor[T, TT] = {
    new Tensor[T, TT](Functions.arange(from, to, new Scalar[Long](1), options))
  }

  def full[T: ClassTag, TT <: TensorType](shape: Shape, fill_value: Scalar[T])(implicit options: TensorOptions[T, TT]): Tensor[T, TT] = {
    new Tensor[T, TT](Functions.full(new IntList(shape.asArray.map(_.toLong)), fill_value))
  }

  def randint[TT <: TensorType](low: Long, high: Long, shape: Shape)(implicit options: TensorOptions[Long, TT]): Tensor[Long, TT] = {
    new Tensor[Long, TT](Functions.randint(low, high, new IntList(shape.asArray.map(_.toLong))))
  }
}
