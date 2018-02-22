package scorch.nn.rnn

import botkop.{numsca => ns}
import scorch.autograd.{Variable, sigmoid, softmax, tanh}

case class GruCell(wir: Variable,
                   bir: Variable,
                   whr: Variable,
                   bhr: Variable,
                   wiz: Variable,
                   biz: Variable,
                   whz: Variable,
                   bhz: Variable,
                   win: Variable,
                   bin: Variable,
                   whn: Variable,
                   bhn: Variable,
                   wy: Variable,
                   by: Variable)
    extends BaseRnnCell(
      Seq(wir, bir, whr, bhr, wiz, biz, whz, bhz, win, bin, whn, bhn, wy, by)) {
  override val na: Int = wir.shape.head
  override val numTrackingStates: Int = 1

  override def forward(xs: Seq[Variable]): Seq[Variable] = xs match {
    case Seq(xt, h0) =>
      val rt = sigmoid(wir.dot(xt) + bir + whr.dot(h0) + bhr)
      val zt = sigmoid(wiz.dot(xt) + biz + whz.dot(h0) + bhz)
      val nt = tanh(win.dot(xt) + bin + rt * (whn.dot(h0) + bhn))
      val ht = (1 - zt) * nt + zt * h0
      val ytHat = softmax(wy.dot(ht) + by)
      Seq(ytHat, ht.detach())
  }
}

object GruCell {
  def apply(na: Int, nx: Int, ny: Int): GruCell = {
    // na : hidden size
    // nx: input size
    // ny: output size

    val i = math.sqrt(2.0 / na)

    val wir = Variable(ns.randn(na, nx) * i, name = Some("wir"))
    val wiz = Variable(ns.randn(na, nx) * i, name = Some("wiz"))
    val win = Variable(ns.randn(na, nx) * i, name = Some("win"))

    val bir = Variable(ns.zeros(na, 1), name = Some("bir"))
    val biz = Variable(ns.zeros(na, 1), name = Some("biz"))
    val bin = Variable(ns.zeros(na, 1), name = Some("bin"))

    val whr = Variable(ns.randn(na, na) * i, name = Some("whr"))
    val whz = Variable(ns.randn(na, na) * i, name = Some("whz"))
    val whn = Variable(ns.randn(na, na) * i, name = Some("whn"))

    val bhr = Variable(ns.zeros(na, 1), name = Some("bhr"))
    val bhz = Variable(ns.zeros(na, 1), name = Some("bhz"))
    val bhn = Variable(ns.zeros(na, 1), name = Some("bhn"))

    val wy = Variable(ns.randn(ny, na) * math.sqrt(2.0 / ny), name = Some("wy"))
    val by = Variable(ns.zeros(ny, 1), name = Some("by"))

    GruCell(wir, bir, whr, bhr, wiz, biz, whz, bhz, win, bin, whn, bhn, wy, by)
  }
}
