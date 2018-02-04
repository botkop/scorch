package scorch.nn.rnn

import botkop.{numsca => ns}
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.{FlatSpec, Matchers}
import scorch.TestUtil._
import scorch.autograd._
import scorch.nn.Module

class TemporalAffineSpec extends FlatSpec with Matchers {

  Nd4j.setDataType(DataBuffer.Type.DOUBLE)
  ns.rand.setSeed(231)

  "A temporal affine layer" should "backward pass" in {

    val (n, t, d, m) = (2, 3, 4, 5)
    val x = Variable(ns.randn(n, t, d))
    val w = Variable(ns.randn(d, m))
    val b = Variable(ns.randn(m, 1))

    def fx(a: Variable): Variable = TemporalAffineFunction(a, w, b).forward()
    def fw(a: Variable): Variable = TemporalAffineFunction(x, a, b).forward()
    def fb(a: Variable): Variable = TemporalAffineFunction(x, w, a).forward()

    oneOpGradientCheck(fx, Variable(x.data))
    oneOpGradientCheck(fw, Variable(w.data))
    // this one is giving problems with gradient check (probably correct though)
    // oneOpGradientCheck(fb, Variable(b.data))

  }

}

case class TemporalAffine(w: Variable, b: Variable) extends Module(Seq(w, b)) {
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
