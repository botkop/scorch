package torch_scala.api.aten

trait TensorType {
   def dtype: Short
}

class CUDA(val index: Short, val dtype: Short) extends TensorType
class CPU(val dtype: Short) extends TensorType


