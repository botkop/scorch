package torch_scala.autograd

import com.typesafe.scalalogging.LazyLogging
import org.scalatest._
import torch_scala.api.aten.{CPU, Shape, Tensor}
import torch_scala.TestUtil._
import torch_scala.api.intsToShape

class FunctionGradientSpec
    extends FlatSpec
    with Matchers
    with BeforeAndAfterEach
    with LazyLogging {

  "Add" should "calculate gradients" in {
    val a = Variable[Double, CPU](Tensor.randn[Double, CPU](Shape(4, 6)))
    val b = Variable[Double, CPU](Tensor.randn[Double, CPU](Shape(4, 6)))
    def f(v1: Variable[Double, CPU], v2: Variable[Double, CPU]): Variable[Double, CPU] = Add(v1, v2).forward()
    binOpGradientCheck(f, a, b)
  }

  it should "calculate gradients with broadcasting" in {
    val a = Variable[Double, CPU](Tensor.randn[Double, CPU](Shape(4, 6)))
    val b = Variable[Double, CPU](Tensor.randn[Double, CPU](Shape(1, 6)))
    def f(a: Variable[Double, CPU], b: Variable[Double, CPU]): Variable[Double, CPU] = Add(a, b).forward()
    binOpGradientCheck(f, a, b)
  }

  "Sub" should "calculate gradients" in {
    val a = Variable[Double, CPU](Tensor.randn[Double, CPU](Shape(3, 4)))
    val b = Variable[Double, CPU](Tensor.randn[Double, CPU](Shape(3, 4)))
    def f(v1: Variable[Double, CPU], v2: Variable[Double, CPU]): Variable[Double, CPU] = Sub(v1, v2).forward()
    binOpGradientCheck(f, a, b)
  }

  it should "calculate gradients with broadcasting" in {
    val a = Variable[Double, CPU](Tensor.randn[Double, CPU](Shape(4, 6)))
    val b = Variable[Double, CPU](Tensor.randn[Double, CPU](Shape(1, 6)))
    def f(a: Variable[Double, CPU], b: Variable[Double, CPU]): Variable[Double, CPU] = Sub(a, b).forward()
    binOpGradientCheck(f, a, b)
  }

  "Mul" should "calculate gradients" in {
    val a = Variable[Double, CPU](Tensor.randn[Double, CPU](Shape(3, 4)))
    val b = Variable[Double, CPU](Tensor.randn[Double, CPU](Shape(3, 4)))
    def f(v1: Variable[Double, CPU], v2: Variable[Double, CPU]): Variable[Double, CPU] = Mul(v1, v2).forward()
    binOpGradientCheck(f, a, b)
  }

  it should "calculate gradients with broadcasting" in {
    val a = Variable[Double, CPU](Tensor.randn[Double, CPU](Shape(4, 6)))
    val b = Variable[Double, CPU](Tensor.randn[Double, CPU](Shape(1, 6)))
    def f(a: Variable[Double, CPU], b: Variable[Double, CPU]): Variable[Double, CPU] = Mul(a, b).forward()
    binOpGradientCheck(f, a, b)
  }

  "Div" should "calculate gradients" in {
    val a = Variable[Double, CPU](Tensor.randn[Double, CPU](Shape(3, 4)))
    val b = Variable[Double, CPU](Tensor.randn[Double, CPU](Shape(3, 4)))
    def f(v1: Variable[Double, CPU], v2: Variable[Double, CPU]): Variable[Double, CPU] = Div(v1, v2).forward()
    binOpGradientCheck(f, a, b)
  }

  it should "calculate gradients with broadcasting" in {
    val a = Variable[Double, CPU](Tensor.randn[Double, CPU](Shape(4, 6)))
    val b = Variable[Double, CPU](Tensor.randn[Double, CPU](Shape(1, 6)))
    def f(a: Variable[Double, CPU], b: Variable[Double, CPU]): Variable[Double, CPU] = Div(a, b).forward()
    binOpGradientCheck(f, a, b)
  }

  "Pow" should "calculate gradients with a const" in {
    val a = Variable[Double, CPU](Tensor.randn[Double, CPU](Shape(4, 6)).abs())
    val b = 3.0
    def f(a: Variable[Double, CPU], b: Double): Variable[Double, CPU] = PowConstant(a, b).forward()
    varConstOpGradientCheck(f, a, b)
  }

  "Matmul" should "calculate gradients" in {
    val a = Variable[Double, CPU](Tensor.randn[Double, CPU](Shape(4, 6)))
    val b = Variable[Double, CPU](Tensor.randn[Double, CPU](Shape(6, 2)))
    def f(a: Variable[Double, CPU], b: Variable[Double, CPU]): Variable[Double, CPU] = Matmul(a, b).forward()
    binOpGradientCheck(f, a, b)
  }

  "Dot" should "calculate gradients" in {
    val a = Variable[Double, CPU](Tensor.randn[Double, CPU](Shape(6)))
    val b = Variable[Double, CPU](Tensor.randn[Double, CPU](Shape(6)))
    def f(a: Variable[Double, CPU], b: Variable[Double, CPU]): Variable[Double, CPU] = Dot(a, b).forward()
    binOpGradientCheck(f, a, b)
  }

  "Exp" should "calculate gradients" in {
    val x = Variable[Double, CPU](Tensor.randn[Double, CPU](Shape(4, 6)))
    def f(a: Variable[Double, CPU]): Variable[Double, CPU] = Exp(a).forward()
    oneOpGradientCheck(f, x)
  }

  "Tanh" should "calculate gradients" in {
    val a = Variable[Double, CPU](Tensor.randn[Double, CPU](Shape(4, 6)))
    def f(a: Variable[Double, CPU]): Variable[Double, CPU] = Tanh(a).forward()
    oneOpGradientCheck(f, a)
  }

  "Cos" should "calculate gradients" in {
    val a = Variable[Double, CPU](Tensor.randn[Double, CPU](Shape(4, 6)))
    def f(a: Variable[Double, CPU]): Variable[Double, CPU] = Cos(a).forward()
    oneOpGradientCheck(f, a)
  }

  "Sin" should "calculate gradients" in {
    val a = Variable[Double, CPU](Tensor.randn[Double, CPU](Shape(4, 6)))
    def f(a: Variable[Double, CPU]): Variable[Double, CPU] = Sin(a).forward()
    oneOpGradientCheck(f, a)
  }

  "Sigmoid" should "calculate gradients" in {
    val a = Variable[Double, CPU](Tensor.randn[Double, CPU](Shape(4, 6)))
    def f(a: Variable[Double, CPU]): Variable[Double, CPU] = Sigmoid(a).forward()
    oneOpGradientCheck(f, a)
  }

  "Softmax" should "calculate gradients" in {
    val a = Variable[Double, CPU](Tensor.randn[Double, CPU](Shape(4, 6)))
    def f(a: Variable[Double, CPU]): Variable[Double, CPU] = Softmax(a, 1).forward()
    oneOpGradientCheck(f, a)
  }

  "Abs" should "calculate gradients" in {
    val a = Variable[Double, CPU](Tensor.randn[Double, CPU](Shape(4, 6)))
    def f(a: Variable[Double, CPU]): Variable[Double, CPU] = Abs(a).forward()
    oneOpGradientCheck(f, a)
  }

  "Sqrt" should "calculate gradients" in {
    val a = Variable[Double, CPU](Tensor.randn[Double, CPU](Shape(4, 6)).abs())
    def f(a: Variable[Double, CPU]): Variable[Double, CPU] = Sqrt(a).forward()
    oneOpGradientCheck(f, a, 1e-6)
  }

  "Mean" should "calculate gradients" in {
    val a = Variable[Double, CPU](Tensor.randn[Double, CPU](Shape(4, 6)))
    def f(a: Variable[Double, CPU]): Variable[Double, CPU] = Mean(a).forward()
    oneOpGradientCheck(f, a)
  }

  "MeanByAxis" should "calculate gradients" in {
    val x = Variable[Double, CPU](Tensor.randn[Double, CPU](Shape(4, 6)))
    def f0(a: Variable[Double, CPU]): Variable[Double, CPU] = MeanByAxis(a, axis = 0).forward()
    oneOpGradientCheck(f0, x)
    val x1 = Variable[Double, CPU](Tensor.randn[Double, CPU](Shape(4, 6)))
    def f1(a: Variable[Double, CPU]): Variable[Double, CPU] = MeanByAxis(a, axis = 1).forward()
    oneOpGradientCheck(f1, x1)
  }

  "Variance" should "calculate gradients" in {
    val a = Variable[Double, CPU](Tensor.randn[Double, CPU](Shape(4, 6)), name = Some("x"))
    def f(a: Variable[Double, CPU]): Variable[Double, CPU] = Variance(a).forward()
    oneOpGradientCheck(f, a, 1e-6)
  }

  "VarianceByAxis" should "calculate gradients" in {
    val x = Variable[Double, CPU](Tensor.randn[Double, CPU](Shape(4, 6)))
    def f0(a: Variable[Double, CPU]): Variable[Double, CPU] = VarianceByAxis(a, axis = 0).forward()
    oneOpGradientCheck(f0, x)
    def f1(a: Variable[Double, CPU]): Variable[Double, CPU] = VarianceByAxis(a, axis = 1).forward()
    oneOpGradientCheck(f1, x.copy())
  }

  "Threshold" should "calculate gradients with a const" in {
    val a = Variable[Double, CPU](Tensor.randn[Double, CPU](Shape(4, 6)).abs())
    val b = 0.0
    def f(a: Variable[Double, CPU], b: Double): Variable[Double, CPU] = Threshold(a, b).forward()
    varConstOpGradientCheck(f, a, b)
  }

  "Max" should "calculate gradients" in {
    val a = Variable[Double, CPU](Tensor.randn[Double, CPU](Shape(4, 6)))
    val b = Variable[Double, CPU](Tensor.randn[Double, CPU](Shape(4, 6)))
    def f(a: Variable[Double, CPU], b: Variable[Double, CPU]): Variable[Double, CPU] = Max(a, b).forward()
    binOpGradientCheck(f, a, b)
  }

//  "Dropout" should "calculate gradients" in {
//    val a = Variable[Double, CPU](Tensor.randn[Double, CPU](Shape(4, 6)))
//
//    val p = 0.5
//    val mask: Tensor[Double, CPU] = (Tensor.randn_like(a.data) lt p).to(a.data) / p
//
//    def f(a: Variable[Double, CPU]): Variable[Double, CPU] =
//      DropoutFunction(a, train = true, maybeMask = Some(mask)).forward()
//    oneOpGradientCheck(f, a)
//  }

  "Concat" should "calculate gradients" in {
    val a = Variable[Double, CPU](Tensor.randn[Double, CPU](Shape(3, 4)))
    val b = Variable[Double, CPU](Tensor.randn[Double, CPU](Shape(4, 4)))
    def f(a: Variable[Double, CPU], b: Variable[Double, CPU]): Variable[Double, CPU] = Concat(a, b).forward()
    binOpGradientCheck(f, a, b)
  }

  it should "calculate gradients along dimension 1" in {
    val a = Variable[Double, CPU](Tensor.randn[Double, CPU](Shape(3, 4)))
    val b = Variable[Double, CPU](Tensor.randn[Double, CPU](Shape(3, 5)))
    def f(a: Variable[Double, CPU], b: Variable[Double, CPU]): Variable[Double, CPU] = Concat(a, b, axis = 1).forward()
    binOpGradientCheck(f, a, b)
  }

}
