package scorch.autograd

import botkop.numsca.Tensor
import botkop.{numsca => ns}
import com.typesafe.scalalogging.LazyLogging
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.factory.Nd4j
import org.scalatest._
import scorch.TestUtil._
import scorch.nn.BatchNorm.BatchNormFunction
import scorch.nn.Dropout.DropoutFunction

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

  "Pow" should "calculate gradients with a const" in {
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

  "Softmax" should "calculate gradients" in {
    val a = Variable(ns.randn(4, 6))
    def f(a: Variable): Variable = Softmax(a).forward()
    oneOpGradientCheck(f, a)
  }

  "Abs" should "calculate gradients" in {
    val a = Variable(ns.randn(4, 6))
    def f(a: Variable): Variable = Abs(a).forward()
    oneOpGradientCheck(f, a)
  }

  "Sqrt" should "calculate gradients" in {
    val a = Variable(ns.randn(4, 6))
    def f(a: Variable): Variable = Sqrt(a).forward()
    oneOpGradientCheck(f, a)
  }

  "Mean" should "calculate gradients" in {
    val a = Variable(ns.randn(4, 6))
    def f(a: Variable): Variable = Mean(a).forward()
    oneOpGradientCheck(f, a)
  }

  "MeanByAxis" should "calculate gradients" in {
    val x = Variable(ns.randn(4, 6))
    def f0(a: Variable): Variable = MeanByAxis(a, axis = 0).forward()
    oneOpGradientCheck(f0, x)
    def f1(a: Variable): Variable = MeanByAxis(a, axis = 1).forward()
    oneOpGradientCheck(f1, x.copy())
  }

  "Variance" should "calculate gradients" in {
    val a = Variable(ns.randn(4, 6))
    def f(a: Variable): Variable = Variance(a).forward()
    oneOpGradientCheck(f, a)
  }

  "VarianceByAxis" should "calculate gradients" in {
    val x = Variable(ns.randn(4, 6))
    def f0(a: Variable): Variable = VarianceByAxis(a, axis = 0).forward()
    oneOpGradientCheck(f0, x)
    def f1(a: Variable): Variable = VarianceByAxis(a, axis = 1).forward()
    oneOpGradientCheck(f1, x.copy())
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

    val p = 0.5
    val mask: Tensor = (ns.rand(a.shape: _*) < p) / p

    def f(a: Variable): Variable =
      DropoutFunction(a, train = true, maybeMask = Some(mask)).forward()
    oneOpGradientCheck(f, a)
  }

  "Concat" should "calculate gradients" in {
    val a = Variable(ns.randn(3, 4))
    val b = Variable(ns.randn(4, 4))
    def f(a: Variable, b: Variable): Variable = Concat(a, b).forward()
    binOpGradientCheck(f, a, b)
  }

  it should "calculate gradients along dimension 1" in {
    val a = Variable(ns.randn(3, 4))
    val b = Variable(ns.randn(3, 5))
    def f(a: Variable, b: Variable): Variable = Concat(a, b, axis = 1).forward()
    binOpGradientCheck(f, a, b)
  }

}
