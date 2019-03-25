package torch_scala.autograd



import torch_scala.api.aten.{Shape, Tensor, TensorOptions, TensorType}
import torch_scala.api.exception.ShapeMismatchException
import torch_scala.nn.Module

import scala.language.implicitConversions
import scala.reflect.ClassTag

case class Variable[T: ClassTag, TT <: TensorType](data: Tensor[T, TT],
                                                   gradFn: Option[Function[T, TT]] = None,
                                                   name: Option[String] = None) {

  override def toString: String =
    if (name.isDefined) s"name: ${name.get}, data: $data ${data.shape}" else s"data: $data ${data.shape}"

  lazy val grad: Variable[T, TT] =
    Variable(Tensor.zeros_like(data), name = name.map(n => s"g_$n"))
  val shape: Shape = data.shape

  def backward(): Unit = {
    backward(Variable(Tensor.ones_like(data)))
  }

  def backward(gradOutput: Variable[T, TT]): Unit = {
    if (!gradOutput.shape.isBroadcastableTo(shape) && gradOutput.shape.rank != 0) {
      throw new ShapeMismatchException(s"${name.getOrElse("")}: gradOutput shape = ${gradOutput.shape}, var shape = ${shape}")
    }
    grad.data += gradOutput.data
    for (gf <- gradFn) gf.backward(gradOutput)
  }

  def zero_grad(): Unit = {
    grad.data *= 0.asInstanceOf[T]
  }

  def detach(name: Option[String] = None) = Variable(data, name = name)

  def T = Transpose(this).forward()

  // chain operator
  def ~>[T1](m: Module[_, TT, T, T1]): Variable[T1, TT] = m.forward(this)
  def ~>[T1](f: (Variable[T, TT]) => Variable[T1, TT]): Variable[T1, TT] = f(this)
}

object Variable {

  def apply[T: ClassTag, TT <: TensorType](name: String)(data: T*)(implicit opt: TensorOptions[T, TT]): Variable[T, TT] =
    new Variable[T, TT](Tensor[T, TT](data.toArray), name = Some(name))
  def apply[T: ClassTag, TT <: TensorType](data: T*)(implicit opt: TensorOptions[T, TT]): Variable[T, TT] =
    new Variable[T, TT](Tensor[T, TT](data.toArray))
  def apply[T: ClassTag, TT <: TensorType](data: Tensor[T, TT], name: String): Variable[T, TT] = new Variable[T, TT](data, name = Some(name))
  def apply[T: ClassTag, TT <: TensorType](data: Tensor[T, TT]): Variable[T, TT] = new Variable[T, TT](data)
}
