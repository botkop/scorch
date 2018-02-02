package scorch.nn

import botkop.numsca.:>
import botkop.{numsca => ns}
import scorch.autograd._

case class Rnn(h0: Variable, wX: Variable, wH: Variable, b: Variable)
    extends Module(Seq(wX, wH, b)) {

  override def forward(x: Variable): Variable =
    RnnFunction(x, h0, wX, wH, b).forward()
}

object Rnn {
  def apply(h0: Variable, d: Int): Rnn = {
    val List(n, h) = h0.shape
    val wX = Variable(ns.randn(d, h) / math.sqrt(d))
    val wH = Variable(ns.randn(h, h) / math.sqrt(h))
    val b = Variable(ns.zeros(h))
    Rnn(h0, wX, wH, b)
  }
}

case class RnnFunction(x: Variable,
                       h0: Variable,
                       wX: Variable,
                       wH: Variable,
                       b: Variable)
    extends Function {
  import RnnFunction._

  val List(n, t, d) = x.shape
  val List(_, h) = h0.shape

  val xs: Seq[Variable] = (0 until t) map { i =>
    Variable(x.data(:>, i, :>).reshape(n, d))
  }

  lazy val vs: List[Variable] = xs.foldLeft(List.empty[Variable]) {
    case (acc, v) =>
      acc :+ stepForward(v, acc.lastOption.getOrElse(h0), wX, wH, b)
  }

  override def forward(): Variable = {
    val data = ns.zeros(n, t, h)
    for (i <- 0 until t) {
      data(:>, i, :>) := vs(i).data
    }
    Variable(data, Some(this))
  }

  override def backward(gradOutput: Variable): Unit = {
    vs.zipWithIndex.reverse.foreach {
      case (v, i) =>
        val gi = Variable(gradOutput.data(:>, i, :>).reshape(v.shape: _*))
        v.backward(gi)
    }

    // still need to update x.grad!!!!!!!!!
    val gData = ns.zerosLike(x.data)
    for (i <- 0 until t) {
      gData(:>, i, :>) := xs(i).g.get
    }
    x.g = Some(gData)
  }

}

object RnnFunction {
  def stepForward(x: Variable,
                  prevH: Variable,
                  wX: Variable,
                  wH: Variable,
                  b: Variable): Variable = {
    val xWx = x dot wX
    val prevHwH = prevH dot wH
    tanh(xWx + prevHwH + b)
  }
}
