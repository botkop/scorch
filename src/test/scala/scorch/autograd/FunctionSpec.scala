package scorch.autograd

import botkop.numsca.Tensor
import botkop.{numsca => ns}
import org.scalatest.{FlatSpec, Matchers}
import scorch.nn.Linear
import scorch._

class FunctionSpec extends FlatSpec with Matchers {

  "Function" should "transpose" in {
    val v = Variable(ns.randn(4, 6))
    val w = Variable(ns.randn(6, 4))

    val vt = v.t()
    vt.shape shouldBe List(6, 4)

    vt.backward(w)
    v.grad.shape shouldBe List(4, 6)
  }

  it should "do a dot product" in {
    val x = Variable(ns.arange(12).reshape(3, 4))
    val y = Variable(ns.arange(8).reshape(4, 2))
    val z = x.dot(y)
    z.shape shouldBe List(3, 2)

    println(z)

    val g = Variable(ns.arange(6).reshape(z.shape.toArray))
    z.backward(g)

    println(x.grad.data)
    println(y.grad.data)

    x.grad.data.data shouldBe Array( //
        1, 3, 5, //
        7, 3, 13, //
        23, 33, 5, //
        23, 41, 59)
    x.grad.shape shouldBe List(3, 4)

    y.grad.data.data shouldBe Array( //
        40, 52, //
        46, 61, //
        52, 70, //
        58, 79)
    y.grad.shape shouldBe List(4, 2)
  }

  it should "do an affine operation" in {

    val inputFeatures = 4
    val outputFeatures = 3
    val numSamples = 16

    def makeVariable(shape: Int*): Variable =
      Variable(ns.arange(shape.product).reshape(shape: _*))

    val w = makeVariable(outputFeatures, inputFeatures)
    val b = makeVariable(1, outputFeatures)

    val x = makeVariable(numSamples, inputFeatures)

    val d = x.dot(w.t())
    d.shape shouldBe List(numSamples, outputFeatures)

    val y = d + b
    y.shape shouldBe d.shape

    val dy = makeVariable(y.shape: _*)
    y.backward(dy)

    println(w.grad.shape)
    println(b.grad)

    println(ns.sum(b.grad.data, axis = 0))

  }

  it should "linear" in {

    val x = ns.arange(12).reshape(4, 3)
    val w = ns.arange(15).reshape(5, 3)
    val b = ns.arange(5)

    val weights = Variable(w, name = Some("weights"))
    val bias = Variable(b, name = Some("bias"))
    val input = Variable(x, name = Some("x"))

    val l = Linear(weights, bias)

    val output: Variable = l.forward(input)

    println(output)

    val dy = Variable(ns.arange(20).reshape(4, 5), name = Some("dy"))

    output.backward(dy)

    println(weights.grad)
    println(bias.grad)

    assert(
      ns.arrayEqual(
        weights.grad.data,
        Tensor(210.0, 240.0, 270.0, 228.0, 262.0, 296.0, 246.0, 284.0, 322.0,
          264.0, 306.0, 348.0, 282.0, 328.0, 374.0).reshape(5, 3)))

    assert(ns.arrayEqual(bias.grad.data, Tensor(30.0, 34.0, 38.0, 42.0, 46.0)))

    /*
    data: tensor of shape List(4, 5) and stride List(5, 1) (20 / 20)
List(5.0, 15.0, 25.0, 35.0, 45.0, 14.0, 51.0, 88.0, 125.0, 162.0, 23.0, 87.0, 151.0, 215.0, 279.0, 32.0, 123.0, 214.0, 305.0, 396.0)
name: g_weights, data: tensor of shape List(5, 3) and stride List(3, 1) (15 / 15)
List(210.0, 240.0, 270.0, 228.0, 262.0, 296.0, 246.0, 284.0, 322.0, 264.0, 306.0, 348.0, 282.0, 328.0, 374.0)
name: g_bias, data: tensor of shape List(5) and stride List(1) (5 / 5)
List(30.0, 34.0, 38.0, 42.0, 46.0)
   */

  }

  it should "handle 2 layers" in {

    val x = Variable(ns.arange(12).reshape(4, 3), name = Some("x"))

    val w1 = Variable(ns.arange(15).reshape(5, 3), name = Some("w1"))
    val b1 = Variable(ns.arange(5), name = Some("b1"))
    val l1 = Linear(w1, b1)

    val w2 = Variable(ns.arange(30).reshape(6, 5), name = Some("w2"))
    val b2 = Variable(ns.arange(6), name = Some("b2"))
    val l2 = Linear(w2, b2)

    val out = x ~> l1 ~> l2
    val dOut = Variable(ns.arange(24).reshape(4, 6))
    out.backward(dOut)

    println(out)
    println(w2.grad)
    println(b2.grad)
    println(w1.grad)
    println(b1.grad)

    /*
data: tensor of shape List(4, 6) and stride List(6, 1) (24 / 24)
List(350.0, 976.0, 1602.0, 2228.0, 2854.0, 3480.0, 1250.0, 3451.0, 5652.0, 7853.0, 10054.0, 12255.0, 2150.0, 5926.0, 9702.0, 13478.0, 17254.0, 21030.0, 3050.0, 8401.0, 13752.0, 19103.0, 24454.0, 29805.0)

name: g_w2, data: tensor of shape List(6, 5) and stride List(5, 1) (30 / 30)
List(936.0, 3564.0, 6192.0, 8820.0, 11448.0, 1010.0, 3840.0, 6670.0, 9500.0, 12330.0, 1084.0, 4116.0, 7148.0, 10180.0, 13212.0, 1158.0, 4392.0, 7626.0, 10860.0, 14094.0, 1232.0, 4668.0, 8104.0, 11540.0, 14976.0, 1306.0, 4944.0, 8582.0, 12220.0, 15858.0)

name: g_b2, data: tensor of shape List(6) and stride List(1) (6 / 6)
List(36.0, 40.0, 44.0, 48.0, 52.0, 56.0)

name: g_w1, data: tensor of shape List(5, 3) and stride List(3, 1) (15 / 15)
List(23850.0, 27650.0, 31450.0, 25632.0, 29708.0, 33784.0, 27414.0, 31766.0, 36118.0, 29196.0, 33824.0, 38452.0, 30978.0, 35882.0, 40786.0)

name: g_b1, data: tensor of shape List(5) and stride List(1) (5 / 5)
List(3800.0, 4076.0, 4352.0, 4628.0, 4904.0)
     */

    assert(
      ns.arrayEqual(
        ns.array(350.0, 976.0, 1602.0, 2228.0, 2854.0, 3480.0, 1250.0, 3451.0,
            5652.0, 7853.0, 10054.0, 12255.0, 2150.0, 5926.0, 9702.0, 13478.0,
            17254.0, 21030.0, 3050.0, 8401.0, 13752.0, 19103.0, 24454.0,
            29805.0)
          .reshape(4, 6),
        out.data
      ))

    assert(
      ns.arrayEqual(
        ns.array(936.0, 3564.0, 6192.0, 8820.0, 11448.0, 1010.0, 3840.0, 6670.0,
            9500.0, 12330.0, 1084.0, 4116.0, 7148.0, 10180.0, 13212.0, 1158.0,
            4392.0, 7626.0, 10860.0, 14094.0, 1232.0, 4668.0, 8104.0, 11540.0,
            14976.0, 1306.0, 4944.0, 8582.0, 12220.0,
            15858.0)
          .reshape(6, 5),
        w2.grad.data
      ))

    assert(
      ns.arrayEqual(
        ns.array(36.0, 40.0, 44.0, 48.0, 52.0, 56.0),
        b2.grad.data
      ))

    assert(
      ns.arrayEqual(
        ns.array(23850.0, 27650.0, 31450.0, 25632.0, 29708.0, 33784.0, 27414.0,
            31766.0, 36118.0, 29196.0, 33824.0, 38452.0, 30978.0, 35882.0,
            40786.0)
          .reshape(5, 3),
        w1.grad.data
      ))

    assert(
      ns.arrayEqual(
        ns.array(3800.0, 4076.0, 4352.0, 4628.0, 4904.0),
        b1.grad.data
      ))

  }

}
