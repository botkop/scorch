package torch_scala.nn

import com.typesafe.scalalogging.LazyLogging
import torch_scala.api.aten.TensorType
import torch_scala.api.types.{IntOrLongOrHalfOrFloatOrDouble, IsIntOrLongOrHalfOrFloatOrDouble}
import torch_scala.autograd.Variable

import scala.language.{higherKinds, implicitConversions}
import scala.reflect.ClassTag

/*
sealed trait Infer[F[_]]
trait LowPriority {
  implicit def inferDefault[F[_]]: Infer[F] = new Infer[F] {}
}
object Infer extends LowPriority {
  type Id[A] = A
  implicit def inferId: Infer[Id] = new Infer[Id] {}
}
 */

abstract class BaseModule[+PT <: Any, TT <: TensorType](localParameters: Seq[Variable[PT, TT]] = Nil) {

  // by default, obtain submodules through introspection
  lazy val subModules: Seq[BaseModule[_ <: Any, TT]] =
    this.getClass.getDeclaredFields.flatMap { f =>
      f setAccessible true
      f.get(this) match {
        case module: BaseModule[_, TT] => Some(module)
        case _                  => None
      }
    }

  def parameters: Seq[Variable[Any, TT]] =
    localParameters.map(_.asInstanceOf[Variable[Any, TT]]) ++ subModules.flatMap(_.parameters)

  def gradients: Seq[Variable[Any, TT]] = parameters.map(_.grad)

  def zeroGrad(): Unit = parameters.foreach(_.zero_grad())

  /*
  Pytorch way of solving distinction between training and test mode is by using a mutable variable.
  Perhaps there is a better way.
   */
  var inTrainingMode: Boolean = false

  /*
  Sets the module in training mode.
  This has any effect only on modules such as Dropout or BatchNorm.
   */
  def train(mode: Boolean = true): Unit = {
    this.inTrainingMode = mode
    subModules.foreach(_.train(mode))
  }

  /*
  Sets the module in evaluation mode.
  This has any effect only on modules such as Dropout or BatchNorm.
   */
  def eval(): Unit = train(false)

}

abstract class Module[+PT <: Any, TT <: TensorType, IT, OT](localParameters: Seq[Variable[PT, TT]] = Nil)
  extends BaseModule[PT, TT](localParameters)
    with LazyLogging {
  def forward(x: Variable[IT, TT]): Variable[OT, TT]
  def apply(x: Variable[IT, TT]): Variable[OT, TT] = forward(x)
//  def par(parallelism: Int = Runtime.getRuntime.availableProcessors / 2): ParallelModule = {
//    logger.info(s"parallelizing factor = $parallelism")
//    ParallelModule(this, parallelism)
//  }
}

abstract class SeqModule[TT <: TensorType](localParameters: Seq[Variable[Any, TT]] = Nil)
  extends BaseModule(localParameters) {
  def forward(xs: Seq[Variable[Any, TT]]): Seq[Variable[Any, TT]]
  def apply(xs: Seq[Variable[Any, TT]]): Seq[Variable[Any, TT]] = forward(xs)
}

/*
abstract class Module[F[_]: Infer](localParameters: Seq[Variable] = Nil)
    extends BaseModule(localParameters) {
  def forward(x: F[Variable]): F[Variable]
  def apply(x: F[Variable]): F[Variable] = forward(x)
}
 */

