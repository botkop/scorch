package scorch.nn.rnn

import botkop.numsca.Tensor
import botkop.{numsca => ns}
import org.scalatest.{FlatSpec, Matchers}
import scorch.autograd.{Variable, _}
import scorch.nn.SGD

import scala.annotation.tailrec
import scala.io.Source
import scala.util.Random

class MyRnnSpec extends FlatSpec with Matchers {

  ns.rand.setSeed(231)

  case class MyRnnCell(wax: Variable,
                       waa: Variable,
                       wya: Variable,
                       ba: Variable,
                       by: Variable)
      extends RecurrentModule(Seq(wax, waa, wya, ba, by)) {

    val List(na, nx) = wax.shape
    val List(ny, _) = wya.shape
    val optimizer = SGD(parameters, lr = 0.01)

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

    def step(): Unit = optimizer.step()
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

  case class Sampler(rnn: MyRnnCell, charToIx: Map[Char, Int], eolIndex: Int) {
    val vocabSize: Int = charToIx.size
    val na: Int = rnn.na
    val ixToChar: Map[Int, Char] = charToIx.map(_.swap)

    @tailrec
    final def generate(counter: Int,
                       prevX: Variable,
                       prevA: Variable,
                       acc: List[Int]): List[Int] =
      if (acc.lastOption.contains(eolIndex)) {
        acc
      } else if (counter >= 50) {
        acc :+ eolIndex
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

    def sample(): String = {
      val x0 = Variable(ns.zeros(vocabSize, 1))
      val a0 = Variable(ns.zeros(na, 1))
      val sampledIndices = generate(1, x0, a0, List.empty[Int])
      sampledIndices.map(ixToChar).mkString
    }
  }

  "My rnn" should "generate dinosaur names" in {

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
    val EolIndex = charToIx('\n') // index for end of line
    val BolIndex = -1 // index for beginning of line

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
                 rnn: MyRnnCell): (Double, Variable) = {
      rnn.zeroGrad()
      val (yHat, a) = rnnForward(xs, aPrev, rnn)
      val loss = rnnLoss(yHat, ys)
      loss.backward()
      rnn.clipGradients(5)
      rnn.step()
      (loss.data.squeeze(), a)
    }

    def model(examples: List[String],
              charToIx: Map[Char, Int],
              numIterations: Int = 35000,
              na: Int = 50,
              numNames: Int = 7,
              vocabSize: Int = 27,
              printEvery: Int = 1000): Unit = {
      val (nx, ny) = (vocabSize, vocabSize)
      val rnn = MyRnnCell(na, nx, ny)

      val sampler = Sampler(rnn, charToIx, EolIndex)

      val aPrev = Variable(ns.zeros(na, 1))

      var totalLoss = 0.0

      for (j <- 1 to numIterations) {
        val index = j % examples.length
        val xs: List[Int] = BolIndex +: examples(index).map(charToIx).toList
        val ys: List[Int] = xs.tail :+ EolIndex

        val (loss, ap) = optimize(xs, ys, aPrev, rnn)
        totalLoss += loss
        aPrev.data := ap.data // seems to have little or no effect. Why?

        if (j % printEvery == 0) {
          println(s"Iteration: $j, Loss: ${totalLoss / printEvery}")
          for (_ <- 1 to numNames) {
            print(sampler.sample())
          }
          println()
          totalLoss = 0.0
        }
      }
    }
    model(examples, charToIx, printEvery = 500)
  }

}
