package scorch.nn.rnn

import botkop.numsca.Tensor
import botkop.{numsca => ns}
import com.typesafe.scalalogging.LazyLogging
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.{FlatSpec, Matchers}
import scorch.autograd.{Variable, _}
import scorch.nn.{Module, Optimizer, SGD}

import scala.annotation.tailrec
import scala.io.Source
import scala.util.Random

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

    def clipGradients(maxValue: Double): Unit = {
      parameters
        .map(_.grad.get)
        .foreach(v => v.data := ns.clip(v.data, -maxValue, maxValue))
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

  case class MyRnnLoss(actuals: Seq[Variable], targets: Seq[Int])
      extends Function {
    override def forward(): Variable = {
      val seqLoss = actuals.zip(targets).foldLeft(0.0) {
        case (loss, (yht, y)) =>
          loss - ns.log(yht.data(y, 0)).squeeze()
      }
      Variable(Tensor(seqLoss), Some(this))
    }

    override def backward(gradOutput: Variable): Unit = {
      actuals.zip(targets).reverse.foreach {
        case (yh, y) =>
          val dy = ns.copy(yh.data)
          dy(y, 0) -= 1
          yh.backward(Variable(dy))
      }
    }
  }

  def rnnLoss(actuals: Seq[Variable], targets: Seq[Int]): Variable =
    MyRnnLoss(actuals, targets).forward()

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

    val examples = Random.shuffle(
      Source
        .fromFile("src/test/resources/dinos.txt")
        .getLines()
        .map(_.toLowerCase)
        .toList)

    val chars = data.toCharArray.distinct.sorted

    val dataSize = data.length
    val vocabSize = chars.length

    println(
      s"There are $dataSize total characters and $vocabSize unique characters in your data")

    val charToIx = chars.zipWithIndex.toMap
    val ixToChar = charToIx.map(_.swap)

    val EolIndex = charToIx('\n')
    val BolIndex = -1

    def sample(rnn: MyRnnCell, charToIx: Map[Char, Int]): List[Int] = {
      @tailrec
      def generate(counter: Int,
                   prevX: Variable,
                   prevA: Variable,
                   acc: List[Int]): List[Int] =
        if (acc.lastOption.contains(EolIndex)) {
          acc
        } else if (counter >= 50) {
          acc :+ EolIndex
        } else {
          val (nextX, nextIdx, nextA) = generateNextChar(prevX, prevA)
          generate(counter + 1, nextX, nextA, acc :+ nextIdx)
        }

      def generateNextChar(xPrev: Variable,
                           aPrev: Variable): (Variable, Int, Variable) = {
        val Seq(yHat, aNext) = rnn(xPrev, aPrev)
        val vocabSize = xPrev.shape.head
        val nextIdx = ns.choice(ns.arange(vocabSize), yHat.data).squeeze().toInt
        val xNext = Variable(ns.zerosLike(xPrev.data))
        xNext.data(nextIdx, 0) := 1
        (xNext, nextIdx, aNext)
      }

      val vocabSize = charToIx.size
      val na = rnn.na

      val x0 = Variable(ns.zeros(vocabSize, 1))
      val a0 = Variable(ns.zeros(na, 1))

      generate(1, x0, a0, List.empty[Int])
    }

    def rnnForward(xs: List[Int],
                   aPrev: Variable,
                   rnn: MyRnnCell,
                   vocabSize: Int = 27): (List[Variable], Variable) =
      xs.foldLeft(List.empty[Variable], aPrev) {
        case ((yHat, a0), x) =>
          val xt = Variable(ns.zeros(vocabSize, 1))
          if (x != BolIndex)
            xt.data(x, 0) := 1
          val Seq(yht, a1) = rnn(xt, a0)
          (yHat :+ yht, a1)
      }

    def optimize(xs: List[Int],
                 ys: List[Int],
                 aPrev: Variable,
                 rnn: MyRnnCell,
                 optimizer: Optimizer): (Double, Variable) = {
      rnn.zeroGrad()
      val (yHat, a) = rnnForward(xs, aPrev, rnn)
      val loss = rnnLoss(yHat, ys)
      loss.backward()
      rnn.clipGradients(5)
      optimizer.step()
      (loss.data.squeeze(), a)
    }

    def model(examples: List[String],
              ixToChar: Map[Int, Char],
              charToIx: Map[Char, Int],
              numIterations: Int = 35000,
              na: Int = 50,
              numNames: Int = 7,
              vocabSize: Int = 27): Unit = {
      val (nx, ny) = (vocabSize, vocabSize)
      val rnn = MyRnnCell(na, nx, ny)
      val optimizer = SGD(rnn.parameters, lr = 0.01)

      val aPrev = Variable(ns.zeros(na, 1))

      var totalLoss = 0.0

      for (j <- 1 to numIterations) {
        val index = j % examples.length
        val xs: List[Int] = BolIndex +: examples(index).map(charToIx).toList
        val ys: List[Int] = xs.tail :+ EolIndex

        val (loss, ap) = optimize(xs, ys, aPrev, rnn, optimizer)
        totalLoss += loss
        aPrev.data := ap.data // seems to have little or no effect. Why?

        val printEvery = 1000

        if (j % printEvery == 0) {
          println(s"Iteration: $j, Loss: ${totalLoss / printEvery}")
          for (_ <- 1 to numNames) {
            val sampledIndices = sample(rnn, charToIx)
            print(sampledIndices.map(ixToChar).mkString)
          }
          println()
          totalLoss = 0.0
        }
      }
    }
    model(examples, ixToChar, charToIx)
  }

}
