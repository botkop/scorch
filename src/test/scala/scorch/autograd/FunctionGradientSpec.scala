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

  Nd4j.setDataType(DataBuffer.Type.DOUBLE)
  ns.rand.setSeed(231)

  override def beforeEach(): Unit = {}

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

  "Pow" should "calculate gradients" in {
    val a = Variable(ns.abs(ns.randn(3, 4)))
    // val b = Variable(ns.abs(ns.randint(3, 3, 4)))
    val b = Variable(2)
    def f(v1: Variable, v2: Variable): Variable = Pow(v1, v2).forward()
    binOpGradientCheck(f, a, b)
  }

  it should "calculate gradients with a const" in {
    val a = Variable(ns.abs(ns.randn(4, 6)))
    val b = 3.0
    def f(a: Variable, b: Double): Variable = PowConstant(a, b).forward()
    varConstOpGradientCheck(f, a, b)
  }

  "Dot" should "calculate gradients" in {
    val a = Variable(ns.randn(4, 6))
    val b = Variable(ns.randn(6, 2))
    def f(a: Variable, b: Variable): Variable = Dot(a, b).forward()
    binOpGradientCheck(f, a, b)
  }

  "Exp" should "calculate gradients" in {
    val a = Variable(ns.randn(4, 6))
    def f(a: Variable): Variable = Exp(a).forward()
    oneOpGradientCheck(f, a)
  }

  "Tanh" should "calculate gradients" in {
    val a = Variable(ns.randn(4, 6))
    def f(a: Variable): Variable = Tanh(a).forward()
    oneOpGradientCheck(f, a)
  }

  "Sigmoid" should "calculate gradients" in {
    val a = Variable(ns.randn(4, 6))
    def f(a: Variable): Variable = Sigmoid(a).forward()
    oneOpGradientCheck(f, a)
  }

  "Mean" should "calculate gradients" in {
    val a = Variable(ns.randn(4, 6))
    def f(a: Variable): Variable = Mean(a).forward()
    oneOpGradientCheck(f, a)
  }

  "Threshold" should "calculate gradients with a const" in {
    val a = Variable(ns.abs(ns.randn(4, 6)))
    val b = 0.0
    def f(a: Variable, b: Double): Variable = Threshold(a, b).forward()
    varConstOpGradientCheck(f, a, b)
  }

  "Max" should "calculate gradients" in {
    val a = Variable(ns.randn(4, 6))
    val b = Variable(ns.randn(4, 6))
    def f(a: Variable, b: Variable): Variable = Max(a, b).forward()
    binOpGradientCheck(f, a, b)
  }

  "Dropout" should "calculate gradients" in {
    val a = Variable(ns.randn(4, 6))
    def f(a: Variable): Variable = DropoutFunction(a, train = false).forward()
    oneOpGradientCheck(f, a)
  }


  def binOpGradientCheck(f: (Variable, Variable) => Variable,
                         a: Variable,
                         b: Variable): Assertion = {

    val out = f(a, b)
    logger.debug(s"out = $out")

    val dOut = Variable(ns.randn(out.shape: _*))
    logger.debug(s"dOut = $dOut")

    out.backward(dOut)

    val da = a.grad.get.data
    val db = b.grad.get.data

    logger.debug(s"da = $da")
    logger.debug(s"db = $db")

    def fa(t: Tensor) = f(Variable(t), b).data
    def fb(t: Tensor) = f(a, Variable(t)).data

    val daNum = evalNumericalGradientArray(fa, a.data, dOut.data)
    val dbNum = evalNumericalGradientArray(fb, b.data, dOut.data)

    logger.debug(s"daNum = $daNum")
    logger.debug(s"dbNum = $dbNum")

    val daError = relError(da, daNum)
    val dbError = relError(db, dbNum)

    logger.debug(s"daError = $daError")
    logger.debug(s"dbError = $dbError")

    assert(daError < 1e-5)
    assert(dbError < 1e-5)
  }

  def varConstOpGradientCheck(f: (Variable, Double) => Variable,
                              a: Variable,
                              b: Double): Assertion = {
    val out = f(a, b)
    logger.debug(s"out = $out")

    val dOut = Variable(ns.randn(out.shape: _*))
    logger.debug(s"dOut = $dOut")

    out.backward(dOut)
    val da = a.grad.get.data
    logger.debug(s"da = $da")

    def fa(t: Tensor) = f(Variable(t), b).data
    val daNum = evalNumericalGradientArray(fa, a.data, dOut.data)
    logger.debug(s"daNum = $daNum")

    val daError = relError(da, daNum)
    logger.debug(s"daError = $daError")

    assert(daError < 1e-5)
  }

  def oneOpGradientCheck(f: (Variable) => Variable,
                         a: Variable): Assertion = {

    val out = f(a)
    logger.debug(s"out = $out")

    val dOut = Variable(ns.randn(out.shape: _*))
    logger.debug(s"dOut = $dOut")

    out.backward(dOut)

    val da = a.grad.get.data
    logger.debug(s"da = $da")

    def fa(t: Tensor) = f(Variable(t)).data

    val daNum = evalNumericalGradientArray(fa, a.data, dOut.data)
    logger.debug(s"daNum = $daNum")

    val daError = relError(da, daNum)
    logger.debug(s"daError = $daError")

    assert(daError < 1e-5)
  }

}
