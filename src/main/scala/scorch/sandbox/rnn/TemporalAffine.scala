package scorch.sandbox.rnn

import botkop.{numsca => ns}
import scorch.autograd._
import scorch.nn.SimpleModule

case class TemporalAffine(w: Variable, b: Variable) extends SimpleModule(Seq(w, b)) {
  override def forward(x: Variable): Variable =
    TemporalAffineFunction(x, w, b).forward()
}

case class TemporalAffineFunction(x: Variable, w: Variable, b: Variable)
    extends Function {

  val List(n, t, d) = x.shape
  val m: Int = w.shape.last

  override def forward(): Variable = {
    val out = x.data.reshape(n * t, d).dot(w.data).reshape(n, t, m) + b.data.T
    Variable(out, Some(this))
  }

  override def backward(gradOutput: Variable): Unit = {
    // dOut = n * t * m

    val dOut = gradOutput.data
    val dx = dOut.reshape(n * t, m).dot(w.data.T).reshape(n, t, d)
    val dw = dOut.reshape(n * t, m).T.dot(x.data.reshape(n * t, d)).T
    val db = ns.sum(ns.sum(dOut, axis = 1), axis = 0).T // todo: fix numsca

    x.backward(Variable(dx))
    w.backward(Variable(dw))
    b.backward(Variable(db))
  }
}
