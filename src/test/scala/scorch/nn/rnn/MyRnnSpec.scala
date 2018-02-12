package scorch.nn.rnn

import botkop.numsca.Tensor
import botkop.{numsca => ns}
import com.typesafe.scalalogging.LazyLogging
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.{FlatSpec, Matchers}
import scorch.autograd.Variable
import scorch.nn.Module
import scorch.autograd._
import scorch.nn._

import scala.collection.mutable.ListBuffer
import scala.io.Source

class MyRnnSpec extends FlatSpec with Matchers {

  Nd4j.setDataType(DataBuffer.Type.DOUBLE)
  ns.rand.setSeed(231)

  abstract class MyModule(localParameters: Seq[Variable] = Nil)
      extends LazyLogging {

    /*
    Pytorch way of solving distinction between training and test mode is by using a mutable variable.
    Perhaps there is a better way.
     */
    var inTrainingMode: Boolean = false

    /*
    Sets the module in training mode.
    This has any effect only on modules such as Dropout or BatchNorm.
     */
    def train(mode: Boolean = true): Unit = {
      this.inTrainingMode = mode
      subModules.foreach(_.train(mode))
    }

    /*
    Sets the module in evaluation mode.
    This has any effect only on modules such as Dropout or BatchNorm.
     */
    def eval(): Unit = train(false)

    def forward(xs: Seq[Variable]): Seq[Variable]
    def apply(xs: Variable*): Seq[Variable] = forward(xs)
    def subModules: Seq[Module] = Seq.empty
    def parameters: Seq[Variable] =
      localParameters ++ subModules.flatMap(_.parameters())
    def zeroGrad(): Unit =
      parameters.flatMap(_.grad).foreach(g => g.data := 0)
  }

  case class MyRnnCell(wax: Variable,
                       waa: Variable,
                       wya: Variable,
                       ba: Variable,
                       by: Variable)
      extends MyModule(Seq(wax, waa, wya, ba, by)) {

    val List(na, nx) = wax.shape
    val List(ny, _) = wya.shape

    override def forward(xs: Seq[Variable]): Seq[Variable] = xs match {
      case Seq(xt, aPrev) =>
        val aNext = tanh(waa.dot(aPrev) + wax.dot(xt) + ba)
        val yt = softmax(wya.dot(aNext) + by)
        Seq(yt, aNext)
    }
  }

  object MyRnnCell {
    def apply(na: Int, nx: Int, ny: Int): MyRnnCell = {
      val wax = Variable(ns.randn(na, nx) * 0.01, name = Some("wax"))
      val waa = Variable(ns.randn(na, na) * 0.01, name = Some("waa"))
      val wya = Variable(ns.randn(ny, na) * 0.01, name = Some("wya"))
      val ba = Variable(ns.zeros(na, 1), name = Some("ba"))
      val by = Variable(ns.zeros(ny, 1), name = Some("by"))
      MyRnnCell(wax, waa, wya, ba, by)
    }
  }

  "My Rnn" should "well dunno...step?" in {
    val na = 5
    val nx = 3
    val ny = 2
    val m = 10
    val rnnStep = MyRnnCell(na, nx, ny)

    val at0 = Variable(ns.zeros(na, m), name = Some("at0"))
    val xt = Variable(ns.randn(nx, m))

    val Seq(yt, at) = rnnStep(xt, at0)

    println(yt)

    val dyt = Variable(ns.randn(yt.shape: _*))
    yt.backward(dyt)

    println(at.grad)
  }

  it should "generate dinosaur names" in {

    val data = Source
      .fromFile("src/test/resources/dinos.txt")
      .mkString
      .toLowerCase

    val chars = data.toCharArray.distinct.sorted

    val dataSize = data.length
    val vocabSize = chars.length

    println(
      s"There are $dataSize total characters and $vocabSize unique characters in your data")

    val charToIx = chars.zipWithIndex.toMap
    val ixToChar = charToIx.map(_.swap)

    def clip(gradients: Seq[Variable], maxValue: Double): Unit = {
      gradients.foreach(v => v.data := ns.clip(v.data, -maxValue, maxValue))
    }

    def sample(rnn: MyRnnCell, charToIx: Map[Char, Int]): ListBuffer[Int] = {
      val vocabSize = charToIx.size
      val na = rnn.na

      val x = Variable(ns.zeros(vocabSize, 1))
      var aPrev = Variable(ns.zeros(na, 1))

      val indices = ListBuffer.empty[Int]
      var idx = -1

      var counter = 0
      val newlineCharacter = charToIx('\n')

      while (idx != newlineCharacter && counter < 50) {
        val Seq(y, a) = rnn(x, aPrev)

        ns.rand.setSeed(counter)

        idx = ns.choice(ns.arange(vocabSize), y.data).squeeze().toInt
        indices.append(idx)

        x.data := ns.zeros(vocabSize, 1)
        x.data(idx, 0) := 1

        aPrev = a
        counter += 1
      }

      if (counter == 50) {
        indices.append(newlineCharacter)
      }
      indices
    }

    val na = 100
    val nx = vocabSize
    val ny = vocabSize
    val rnn = MyRnnCell(na, nx, ny)

    val indices = sample(rnn, charToIx)
    println(indices.map(ixToChar).mkString)

  }

}
