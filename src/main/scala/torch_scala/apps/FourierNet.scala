package torch_scala.apps

import org.bytedeco.javacpp.{FloatPointer, IntPointer, Loader, LongPointer}
import org.bytedeco.javacpp.tools.{Builder, Generator, Logger}
import torch_scala.api.aten.functions.Functions.Deallocator_Pointer
import torch_scala.api._
import torch_scala.api.aten._
import torch_scala.api.nn.Module
import torch_scala.examples.FourierNet
import torch_scala.api._
import torch_scala.api.aten.functions.{Basic, Functions}
import torch_scala.api.aten.functions.Math._
import torch_scala.api.aten.functions.Basic._
import torch_scala.autograd.Variable



object FourierNetApp extends App {


  //System.setProperty("org.bytedeco.javacpp.loadlibraries", "false")

//  val net = new FourierNet(20)
//  val pred = net.train(Array[Float](1, 2, 3, 3, 4, 5, 6, 7, 6, 4, 3, 2, 4, 4, 5), 500, Array[Float](1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1))
//  val loss = net.loss(pred, Array[Float](1, 2, 3, 3, 4, 5, 6, 7, 6, 4, 3, 2, 4, 4, 5), Array[Float](1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1))
//  System.out.println("loss = " + loss)

  val data = Array(4l, 5l)
  val list = new IntList(data)

  println(list.data().get(1))

  val t = Tensor.cpu[Float](Array(2f,3f,5f,6f), Shape(4))


  t.put(0l, 55f)

  println(t.data_with_shape())

  println(Tensor.summarize(t, 5))

  t.dot(t)

  val v = Variable[Double, CPU](Tensor.randn[Double, CPU](Shape(4, 6)))
//
//  println(t.toString, t.dim, t.scalar_type())
//  println(t.data.mkString(","))


  val arrayRef = new ArrayRefFloat(Array(1f,4f,6f))

//  val t1 = Functions.tensor[CudaTensorType[Float]](arrayRef)
//  println(t1.toString, t1.dim, t1.scalar_type())

//  println(t1.cpu().data().mkString(", "))

  val dev = CudaDevice
  println(dev.has_index(), dev.is_cuda())

  val d = Array(3, 10, 4, 6, 3, 10, 4, 6, 3, 10, 4, 6, 3, 10, 4, 6)
  val t2 = Tensor.cpu[Int](d, Shape(4, 4)) + 4

  Basic.cat(t2, t2, 0)


  println(Tensor.summarize(Basic.cat(t2, t2, t2)(0), 5))

  val t3 = Tensor.cpu(Array[Long](3, 10, 4, 6)).reshape(Shape(2,2))

  println(t3.+(t3).scalar_type())




}