package scorch.autograd

import botkop.numsca.Tensor
import botkop.{numsca => ns}
import com.typesafe.scalalogging.LazyLogging
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.factory.Nd4j
import org.scalatest._
import scorch.TestUtil._

class FunctionGradientSpec
    extends FlatSpec
    with Matchers
    with BeforeAndAfterEach
    with LazyLogging {

  override def beforeEach(): Unit = {
    Nd4j.setDataType(DataBuffer.Type.DOUBLE)
  }

  "Add" should "calculate gradients" in {
    val a = Variable(ns.randn(4, 6))
    val b = Variable(ns.randn(4, 6))
    def f(v1: Variable, v2: Variable): Variable = Add(v1, v2).forward()
    binOpGradientCheck(f, a, b)
  }

  it should "calculate gradients with broadcasting" in {
    val a = Variable(ns.randn(4, 6))
    val b = Variable(ns.randn(1, 6))
    def f(a: Variable, b: Variable): Variable = Add(a, b).forward()
    binOpGradientCheck(f, a, b)
  }

  "Sub" should "calculate gradients" in {
    val a = Variable(ns.randn(3, 4))
    val b = Variable(ns.randn(3, 4))
    def f(v1: Variable, v2: Variable): Variable = Sub(v1, v2).forward()
    binOpGradientCheck(f, a, b)
  }

  it should "calculate gradients with broadcasting" in {
    val a = Variable(ns.randn(4, 6))
    val b = Variable(ns.randn(1, 6))
    def f(a: Variable, b: Variable): Variable = Sub(a, b).forward()
    binOpGradientCheck(f, a, b)
  }

  "Mul" should "calculate gradients" in {
    val a = Variable(ns.randn(3, 4))
    val b = Variable(ns.randn(3, 4))
    def f(v1: Variable, v2: Variable): Variable = Mul(v1, v2).forward()
    binOpGradientCheck(f, a, b)
  }

  it should "calculate gradients with broadcasting" in {
    val a = Variable(ns.randn(4, 6))
    val b = Variable(ns.randn(1, 6))
    def f(a: Variable, b: Variable): Variable = Mul(a, b).forward()
    binOpGradientCheck(f, a, b)
  }

  "Div" should "calculate gradients" in {
    val a = Variable(ns.randn(3, 4))
    val b = Variable(ns.randn(3, 4))
    def f(v1: Variable, v2: Variable): Variable = Div(v1, v2).forward()
    binOpGradientCheck(f, a, b)
  }

  it should "calculate gradients with broadcasting" in {
    val a = Variable(ns.randn(4, 6))
    val b = Variable(ns.randn(1, 6))
    def f(a: Variable, b: Variable): Variable = Div(a, b).forward()
    binOpGradientCheck(f, a, b)
  }

  "Dot" should "calculate gradients" in {
    val a = Variable(ns.randn(4, 6))
    val b = Variable(ns.randn(6, 2))
    def f(a: Variable, b: Variable): Variable = Dot(a, b).forward()
    binOpGradientCheck(f, a, b)
  }

  def binOpGradientCheck(f: (Variable, Variable) => Variable,
                         a: Variable,
                         b: Variable): Assertion = {

    val out = f(a, b)

    val dOut = Variable(ns.randn(out.shape: _*))
    out.backward(dOut)

    val da = a.grad.get.data
    val db = b.grad.get.data

    def fa(t: Tensor) = f(Variable(t), b).data
    def fb(t: Tensor) = f(a, Variable(t)).data

    val daNum = evalNumericalGradientArray(fa, a.data, dOut.data)
    val dbNum = evalNumericalGradientArray(fb, b.data, dOut.data)

    val daError = relError(da, daNum)
    val dbError = relError(db, dbNum)

    assert(daError < 1e-8)
    assert(dbError < 1e-8)
  }

}
