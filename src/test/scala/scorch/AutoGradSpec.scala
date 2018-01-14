package scorch

import scorch.autograd._
import botkop.numsca.Tensor
import botkop.{numsca => ns}
import org.scalactic.{Equality, TolerantNumerics}
import org.scalatest.{FlatSpec, Matchers}

class AutoGradSpec extends FlatSpec with Matchers {

  "Autograd" should "calculate the gradient" in {

    val x = Variable(-2)
    val y = Variable(5)
    val z = Variable(-4)

    val q = x + y

    val f = q * z

    val df = Variable(1)
    f.backward(df)

    println(x.grad)
    println(y.grad)
    println(z.grad)
    println(q.grad)
    println(f.grad)

    assert(x.grad.get.data.squeeze() == -4)
    assert(y.grad.get.data.squeeze() == -4)
    assert(z.grad.get.data.squeeze() == 3)
    assert(q.grad.get.data.squeeze() == -4)
    assert(f.grad.get.data.squeeze() == 1)

  }

  it should "do sigmoid backward" in {

    val w0 = Variable(2)
    val x0 = Variable(-1)
    val w1 = Variable(-3)
    val x1 = Variable(-2)
    val w2 = Variable(-3)

    // forward pass
    val dot = w0 * x0 + w1 * x1 + w2

    val out = 1 / (1 + exp(-dot))
    out.backward()

    println(w0.grad)
    println(x0.grad)
    println(w1.grad)
    println(x1.grad)
    println(w2.grad)

    implicit val doubleEquality: Equality[Double] =
      TolerantNumerics.tolerantDoubleEquality(0.01)

    assert(w0.grad.get.data.squeeze() === -0.2)
    assert(x0.grad.get.data.squeeze() === 0.39)
    assert(w1.grad.get.data.squeeze() === -0.39)
    assert(x1.grad.get.data.squeeze() === -0.59)
    assert(w2.grad.get.data.squeeze() === 0.2)

  }

  it should "derive constants as 1" in {
    val x = Variable(3)
    x.backward()
    assert(x.grad.get.data.squeeze() == 1)

    val y = Variable(ns.full(Array(3, 3), -2))
    y.backward()
    assert(ns.arrayEqual(y.grad.get.data, ns.ones(3, 3)))

    val z = Variable(ns.zeros(3, 3))
    z.backward()
    assert(ns.arrayEqual(z.grad.get.data, ns.ones(3, 3)))
  }

  it should "derive multiplication with a constant" in {
    val x = Variable(3)
    val y = x * 3
    y.backward()
    assert(x.grad.get.data.squeeze() == 3)
  }

  it should "derive multiplication with itself" in {
    val x = Variable(3)
    val y = x * x
    y.backward()
    assert(x.grad.get.data.squeeze() == 6)
  }

  it should "derive square" in {
    val x = Variable(3)
    val y = x ** 2
    y.backward()
    assert(x.grad.get.data.squeeze() == 6)
  }

  it should "derive division with a constant" in {
    implicit val doubleEquality: Equality[Double] =
      TolerantNumerics.tolerantDoubleEquality(0.01)

    val x = Variable(3)
    val y = x / 3
    y.backward()
    assert(x.grad.get.data.squeeze() === 0.33)
  }

  it should "derive the mean" in {
    val x = Variable(ns.ones(2, 2))
    val y = x + 2
    val z = y * y * 3
    val out = z.mean()
    out.backward()
    println(x.grad.get.data)
    assert(ns.arrayEqual(x.grad.get.data, ns.full(x.data.shape, 4.5)))
  }

  it should "do crazy stuff" in {
    val x = Variable(ns.ones(3, 1))
    val y = x * 2
    def acc(v: Variable): Variable = if (ns.sum(v.data) < 100) acc(v * 2) else v
    val z = acc(y)
    z.backward(Variable(Tensor(0.1, 1.0, 0.0001)))
    println(x.grad)
    assert(ns.arrayEqual(x.grad.get.data, Tensor(6.4, 64, 0.0064)))
  }

}
