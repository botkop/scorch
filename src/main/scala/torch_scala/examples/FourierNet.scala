package torch_scala.examples

import org.bytedeco.javacpp._
import org.bytedeco.javacpp.annotation._
import torch_scala.api.nn.Module


@Platform(include = Array("torch/all.h", "models/FourierNet.h", "<iostream>", "<vector>", "<map>"))
class FourierNet(val size: Int) extends Module {
  allocate(size)

  @native def allocate(@Cast(Array("int")) size: Int): Unit

  @StdVector @native private def train(@StdVector data: FloatPointer, steps: Int, @StdVector weights: FloatPointer): FloatPointer

  def train(data: Array[Float], steps: Int, weights: Array[Float]): Array[Float] = {
    val res = train(new FloatPointer(data:_*), steps, new FloatPointer(weights:_*))
    Array.range(0, data.length).map(i => res.get(i))
  }

  def loss(y1: Array[Float], y2: Array[Float], weights: Array[Float]): Float = {

    y1.zip(y2).map(p => p._1 - p._2).map(x => x*x).zip(weights).map(p => p._1 * p._2).sum

  }
}

