package torch_scala.aten

import org.scalatest.{FlatSpec, Matchers}
import torch_scala.api.aten.{CPU, CUDA, Shape, Tensor}
import torch_scala.api.types._

class TensorSpec extends FlatSpec with Matchers {

  "CPU Tensor" should "be created" in {
    val t1 = Tensor.arange[Long, CPU](-5, 8)
    t1.shape.asArray shouldBe Array(13)
    t1.data() shouldBe Array.range(-5, 8)

    val t2 = Tensor.arange[Double, CPU](-5, 8)
    t2.data() shouldBe Array.range(-5, 8).map(_.toDouble)

    val t3 = Tensor.zeros[Float, CPU](Shape(2, 3))
    t3.data() shouldBe Array.fill(6)(0f)

    val t4 = Tensor.zeros[Int, CPU](Shape(2, 3))
    t4.data() shouldBe Array.fill(6)(0)

    val t5 = Tensor.apply[Byte, CPU](Array.fill(6)(1.toByte)).reshape(Shape(2, 3))
    t5.data() shouldBe Array.fill(6)(1.toByte)

    t1.dataType shouldBe INT64
    t2.dataType shouldBe FLOAT64
    t3.dataType shouldBe FLOAT32
    t4.dataType shouldBe INT32

    // from blob
    val t6 = Tensor.cpu[Byte](Array.fill(6)(1.toByte), Shape(2, 3))
    t6.data() shouldBe Array.fill(6)(1.toByte)

  }


  "CUDA Tensor" should "be created" in {
    val t1 = Tensor.arange[Long, CUDA](-5, 8)
    t1.shape.asArray shouldBe Array(13)
    t1.cpu().data() shouldBe Array.range(-5, 8)

    val t2 = Tensor.arange[Double, CUDA](-5, 8)
    t2.cpu().data() shouldBe Array.range(-5, 8).map(_.toDouble)

    val t3 = Tensor.zeros[Float, CUDA](Shape(2, 3))
    t3.cpu().data() shouldBe Array.fill(6)(0f)

    val t4 = Tensor.zeros[Int, CUDA](Shape(2, 3))
    t4.cpu().data() shouldBe Array.fill(6)(0)

    val t5 = Tensor.apply[Byte, CUDA](Array.fill(6)(1.toByte)).reshape(Shape(2, 3))
    t5.cpu().data() shouldBe Array.fill(6)(1.toByte)

    t1.dataType shouldBe INT64
    t2.dataType shouldBe FLOAT64
    t3.dataType shouldBe FLOAT32
    t4.dataType shouldBe INT32
    t5.dataType shouldBe INT8

    t1.tensorType.isInstanceOf[CUDA] shouldBe true

  }

}
