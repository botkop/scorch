package scorch.nn

import botkop.{numsca => ns}
import com.typesafe.scalalogging.LazyLogging
import scorch.autograd.Variable

import scala.collection.parallel.ParSeq

case class ParallelizeModule(module: Module, parallelism: Int) extends Module {

  import ParallelizeModule._
  override def forward(x: Variable): Variable =
    ParallelizeFunction(x, module, parallelism).forward()
}

object ParallelizeModule {

  case class ParallelizeFunction(x: Variable, module: Module, parallelism: Int)
      extends scorch.autograd.Function
      with LazyLogging {
    import ns._

    val batchSize: Int = x.shape.head
    val chunkSize: Int = batchSize / parallelism

    val fromTos: ParSeq[(Int, Int)] = (0 until batchSize)
      .sliding(chunkSize, chunkSize)
      .map(s => (s.head, s.last + 1))
      .toSeq
      .par

    lazy val (xs, predictions) =
      fromTos.map {
        case (first, last) =>
          val cx = Variable(x.data(first :> last))
          (cx, module(cx))
      }.unzip

    override def forward(): Variable =
      Variable(ns.concatenate(predictions.map(_.data).seq), Some(this))

    override def backward(gradOutput: Variable): Unit = {
      predictions.zip(fromTos).foreach {
        case (v, (first, last)) =>
          val g = Variable(gradOutput.data(first :> last))
          v.backward(g)
      }

      val gradient = Variable(ns.concatenate(xs.map(_.grad.data).seq))
      x.backward(gradient)
    }
  }
}
