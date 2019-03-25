package torch_scala.autograd

import org.scalactic.{Equality, TolerantNumerics}
import org.scalatest.{FlatSpec, Matchers}
import torch_scala.api.aten.{CPU, Shape, Tensor}
import torch_scala.api.aten.functions.Math._
import torch_scala.api.{doubleToScalar, intToScalar}
import torch_scala.autograd.MathVariable._

class AutoGradSpec extends FlatSpec with Matchers {

  "Autograd" should "calculate the gradient" in {

    val x = Variable[Int, CPU]("x")(-2)
    val y = Variable[Int, CPU]("y")(5)
    val z = Variable[Int, CPU]("z")(-4)

    val q = x + y

    val f = q * z

    val df = Variable[Int, CPU](1)
    f.backward(df)

    println(x)
    println(y)
    println(z)
    println(q)
    println(f)

    assert(x.grad.data.item() == -4)
    assert(y.grad.data.item() == -4)
    assert(z.grad.data.item() == 3)
    assert(q.grad.data.item() == -4)
    assert(f.grad.data.item() == 1)

  }

  it should "do sigmoid backward" in {

    val w0 = Variable[Double, CPU](2)
    val x0 = Variable[Double, CPU](-1)
    val w1 = Variable[Double, CPU](-3)
    val x1 = Variable[Double, CPU](-2)
    val w2 = Variable[Double, CPU](-3)

    // forward pass
    val dot: Variable[Double, CPU] = w0 * x0 + w1 * x1 + w2

    val out: Variable[Double, CPU] = 1.0 / ((-dot).exp() + 1)
    out.backward()

    println(w0.grad)
    println(x0.grad)
    println(w1.grad)
    println(x1.grad)
    println(w2.grad)

    implicit val doubleEquality: Equality[Double] =
      TolerantNumerics.tolerantDoubleEquality(0.01)

    assert(w0.grad.data.item() === -0.2)
    assert(x0.grad.data.item() === 0.39)
    assert(w1.grad.data.item() === -0.39)
    assert(x1.grad.data.item() === -0.59)
    assert(w2.grad.data.item() === 0.2)

  }

  it should "derive constants as 1" in {
    val x = Variable[Int, CPU](3)
    x.backward()
    assert(x.grad.data.item() == 1)

    val y = Variable(Tensor.full[Int, CPU](Shape(3, 3), -2))
    y.backward()
    val data: Array[Array[Int]] = y.grad.data.data_with_shape().map(_.asInstanceOf[Array[Int]])
    assert(y.grad.data.data() sameElements Array.fill(3 * 3)(1))

    val z = Variable(Tensor.zeros[Int, CPU](Shape(3, 3)))
    z.backward()
    assert(z.grad.data.data() sameElements  Array.fill(3 * 3)(1))
  }

  it should "derive multiplication with a constant" in {
    val x = Variable[Double, CPU](3)
    val y = x * 3
    y.backward()
    assert(x.grad.data.item() == 3)
  }

  it should "derive multiplication with itself" in {
    val x = Variable[Double, CPU](3)
    val y = x * x
    y.backward()
    assert(x.grad.data.item() == 6)
  }

  it should "derive square" in {
    val x = Variable[Double, CPU](3)
    val y = x ** 2
    y.backward()
    assert(x.grad.data.item() == 6)
  }

  it should "derive division with a constant" in {
    implicit val doubleEquality: Equality[Double] =
      TolerantNumerics.tolerantDoubleEquality(0.01)

    val x = Variable[Double, CPU](3)
    val y = x / 3
    y.backward()
    assert(x.grad.data.item() === 0.33)
  }

  it should "derive the mean" in {
    val x = new Variable[Double, CPU](Tensor.ones(Shape(2, 2)))
    val y = x + 2
    val z = y * y * 3
    val out = z.mean()
    out.backward()
    println(x.grad.data)
    assert(x.grad.data.shape.asArray sameElements Array(2, 2))
    assert(x.grad.data.data() sameElements Array.fill(2 * 2)(4.5))
  }

  it should "do crazy stuff" in {
    val x = new Variable[Double, CPU](Tensor.ones(Shape(3, 1)))
    val y = x * 2
    def acc(v: Variable[Double, CPU]): Variable[Double, CPU] = if (v.data.sum(0, 1).item() < 100) acc(v * 2) else v
    val z = acc(y)
    z.backward(new Variable(Tensor[Double, CPU](0.1, 1.0, 0.0001).reshape(Shape(3, 1))))
    println(x.grad)
    assert(x.grad.data.data() sameElements Array(6.4, 64, 0.0064) )
  }

  it should "derive mse" in {
    val nOut = 4
    val minibatch = 3

    val input = new Variable[Double, CPU](Tensor.randn(Shape(minibatch, nOut)))
    val label = new Variable[Double, CPU](Tensor.randn(Shape(minibatch, nOut)))

    val diff = input - label
    val sqDiff = diff * diff
    val msePerEx = sqDiff.mean()
    val avgMSE = msePerEx.mean().reshape(Shape(1, 1))

    avgMSE.shape shouldBe Shape(1, 1)

    avgMSE.backward()

    input.grad.shape shouldBe input.shape

  }

}
