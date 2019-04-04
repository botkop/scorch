package torch_scala.autograd

import org.scalatest.{FlatSpec, Matchers}
import torch_scala.api.aten.{CPU, Shape, Tensor}

class FunctionSpec extends FlatSpec with Matchers {

  "Function" should "transpose" in {
    val v = Variable(Tensor.randn[Double, CPU](Shape(4, 6)))
    val w = Variable(Tensor.randn[Double, CPU](Shape(6, 4)))

    val vt = v.T
    vt.shape shouldBe Shape(6, 4)

    vt.backward(w)
    v.grad.shape shouldBe Shape(4, 6)
  }

  it should "do a dot product" in {
    val x = Variable(Tensor.arange[Double, CPU](0, 12).reshape(Shape(3, 4)))
    val y = Variable(Tensor.arange[Double, CPU](0, 8).reshape(Shape(4, 2)))
    val z: Variable[Double, CPU] = Matmul(x, y).forward()
    z.shape shouldBe Shape(3, 2)

    println(z)

    val g = Variable(Tensor.arange[Double, CPU](0, 6).reshape(z.shape))
    z.backward(g)

    println(x.grad.data)
    println(y.grad.data)

    x.grad.data.data() shouldBe Array( //
        1, 3, 5, //
        7, 3, 13, //
        23, 33, 5, //
        23, 41, 59)
    x.grad.shape shouldBe Shape(3, 4)

    y.grad.data.data shouldBe Array( //
        40, 52, //
        46, 61, //
        52, 70, //
        58, 79)
    y.grad.shape shouldBe Shape(4, 2)
  }




}
