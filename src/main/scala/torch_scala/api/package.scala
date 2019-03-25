package torch_scala

import torch_scala.api.aten.PrimitivePointer.PrimitivePointer
import torch_scala.api.aten._
import torch_scala.exceptions.InvalidArgumentException

package object api {

  type Indexer = aten.Indexer
  type Index = aten.Index
  type Slice = aten.Slice


  val ---    : Indexer = aten.Ellipsis
  val NewAxis: Indexer = aten.NewAxis
  val ::     : Slice   = aten.Slice.::


  implicit def intToIndex(index: Int): Index = Index(index = index)

  implicit def intToIndexerConstruction(n: Int): IndexerConstructionWithOneNumber = {
    IndexerConstructionWithOneNumber(n)
  }

  implicit def primitiveToScalar[T](v: T): Scalar[T] = new Scalar[T](v)
  implicit def intToScalar(v: Int): Scalar[Int] = new Scalar(v)
  implicit def doubleToScalar(v: Double): Scalar[Double] = new Scalar(v)
  implicit def floatToScalar(v: Float): Scalar[Float] = new Scalar(v)
  implicit def longToScalar(v: Long): Scalar[Long] = new Scalar(v)

  implicit def intsToShape(is: Int*): Shape = Shape(is.toArray)

  private[api] trait API {

    type CancelledException = exceptions.CancelledException
    type UnknownException = exceptions.UnknownException
    type InvalidArgumentException = exceptions.InvalidArgumentException
    type DeadlineExceededException = exceptions.DeadlineExceededException
    type NotFoundException = exceptions.NotFoundException
    type AlreadyExistsException = exceptions.AlreadyExistsException
    type PermissionDeniedException = exceptions.PermissionDeniedException
    type UnauthenticatedException = exceptions.UnauthenticatedException
    type ResourceExhaustedException = exceptions.ResourceExhaustedException
    type FailedPreconditionException = exceptions.FailedPreconditionException
    type AbortedException = exceptions.AbortedException
    type OutOfRangeException = exceptions.OutOfRangeException
    type UnimplementedException = exceptions.UnimplementedException
    type InternalException = exceptions.InternalException
    type UnavailableException = exceptions.UnavailableException
    type DataLossException = exceptions.DataLossException

    val CancelledException         : exceptions.CancelledException.type          = exceptions.CancelledException
    val UnknownException           : exceptions.UnknownException.type            = exceptions.UnknownException
    val InvalidArgumentException   : exceptions.InvalidArgumentException.type    = exceptions.InvalidArgumentException
    val DeadlineExceededException  : exceptions.DeadlineExceededException.type   = exceptions.DeadlineExceededException
    val NotFoundException          : exceptions.NotFoundException.type           = exceptions.NotFoundException
    val AlreadyExistsException     : exceptions.AlreadyExistsException.type      = exceptions.AlreadyExistsException
    val PermissionDeniedException  : exceptions.PermissionDeniedException.type   = exceptions.PermissionDeniedException
    val UnauthenticatedException   : exceptions.UnauthenticatedException.type    = exceptions.UnauthenticatedException
    val ResourceExhaustedException : exceptions.ResourceExhaustedException.type  = exceptions.ResourceExhaustedException
    val FailedPreconditionException: exceptions.FailedPreconditionException.type = exceptions.FailedPreconditionException
    val AbortedException           : exceptions.AbortedException.type            = exceptions.AbortedException
    val OutOfRangeException        : exceptions.OutOfRangeException.type         = exceptions.OutOfRangeException
    val UnimplementedException     : exceptions.UnimplementedException.type      = exceptions.UnimplementedException
    val InternalException          : exceptions.InternalException.type           = exceptions.InternalException
    val UnavailableException       : exceptions.UnavailableException.type        = exceptions.UnavailableException
    val DataLossException          : exceptions.DataLossException.type           = exceptions.DataLossException

    type ShapeMismatchException = exception.ShapeMismatchException
    type GraphMismatchException = exception.GraphMismatchException
    type IllegalNameException = exception.IllegalNameException
    type InvalidDeviceException = exception.InvalidDeviceException
    type InvalidShapeException = exception.InvalidShapeException
    type InvalidIndexerException = exception.InvalidIndexerException
    type InvalidDataTypeException = exception.InvalidDataTypeException
    type OpBuilderUsedException = exception.OpBuilderUsedException
    type CheckpointNotFoundException = exception.CheckpointNotFoundException

    val ShapeMismatchException     : exception.ShapeMismatchException.type      = exception.ShapeMismatchException
    val GraphMismatchException     : exception.GraphMismatchException.type      = exception.GraphMismatchException
    val IllegalNameException       : exception.IllegalNameException.type        = exception.IllegalNameException
    val InvalidDeviceException     : exception.InvalidDeviceException.type      = exception.InvalidDeviceException
    val InvalidShapeException      : exception.InvalidShapeException.type       = exception.InvalidShapeException
    val InvalidIndexerException    : exception.InvalidIndexerException.type     = exception.InvalidIndexerException
    val InvalidDataTypeException   : exception.InvalidDataTypeException.type    = exception.InvalidDataTypeException
    val OpBuilderUsedException     : exception.OpBuilderUsedException.type      = exception.OpBuilderUsedException
    val CheckpointNotFoundException: exception.CheckpointNotFoundException.type = exception.CheckpointNotFoundException
  }

  object exception {


    case class ShapeMismatchException(message: String = null, cause: Throwable = null)
      extends InvalidArgumentException(message, cause)

    case class GraphMismatchException(message: String = null, cause: Throwable = null)
      extends InvalidArgumentException(message, cause)

    case class IllegalNameException(message: String = null, cause: Throwable = null)
      extends InvalidArgumentException(message, cause)

    case class InvalidDeviceException(message: String = null, cause: Throwable = null)
      extends InvalidArgumentException(message, cause)

    case class InvalidShapeException(message: String = null, cause: Throwable = null)
      extends InvalidArgumentException(message, cause)

    case class InvalidIndexerException(message: String = null, cause: Throwable = null)
      extends InvalidArgumentException(message, cause)

    case class InvalidDataTypeException(message: String = null, cause: Throwable = null)
      extends InvalidArgumentException(message, cause)

    case class OpBuilderUsedException(message: String = null, cause: Throwable = null)
      extends InvalidArgumentException(message, cause)

    case class CheckpointNotFoundException(message: String = null, cause: Throwable = null)
      extends exceptions.TorchException(message, cause)
  }
}