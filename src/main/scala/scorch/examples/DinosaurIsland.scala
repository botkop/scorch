package scorch.examples

import botkop.numsca.Tensor
import botkop.{numsca => ns}
import scorch.autograd.{Function, Variable, softmax, tanh}
import scorch.nn.SGD
import scorch.nn.rnn.RecurrentModule

import scala.annotation.tailrec
import scala.io.Source
import scala.util.Random

/**
  * Dinosaurus Island -- Character level language model
  */
object DinosaurIsland extends App {
  ns.rand.setSeed(231)

  val examples = Random.shuffle(
    Source
      .fromFile("src/test/resources/dinos.txt")
      .getLines()
      .map(_.toLowerCase)
      .toList)

  val chars = examples.mkString.toCharArray.distinct.sorted :+ '\n'
  val vocabSize = chars.length
  val charToIx = chars.zipWithIndex.toMap
  val EolIndex = charToIx('\n') // index for end of line
  val BolIndex = -1 // index for beginning of line

  /**
    * Convenience method for computing the loss.
    * Instantiates a LossFunction object, and applies it
    * @param actuals source for the loss function
    * @param targets targets to compute the loss against
    * @return the loss variable, which can be backpropped into
    */
  def rnnLoss(actuals: Seq[Variable], targets: Seq[Int]): Variable =
    LossFunction(actuals, targets).forward()

  /**
    *
    * @param xs
    * @param aPrev
    * @param rnn
    * @param vocabSize
    * @return
    */
  def rnnForward(xs: List[Int],
                 aPrev: Variable,
                 rnn: RnnCell,
                 vocabSize: Int = 27): (List[Variable], Variable) =
    xs.foldLeft(List.empty[Variable], aPrev) {
      case ((yHat, a0), x) =>
        val xt = Variable(ns.zeros(vocabSize, 1))
        if (x != BolIndex)
          xt.data(x, 0) := 1
        val Seq(yht, a1) = rnn(xt, a0)
        (yHat :+ yht, a1)
    }

  /**
    * Execute one step of the optimization to train the model
    * @param xs list of integers, where each integer is a number that maps to a character in the vocabulary
    * @param ys list of integers, exactly the same as xs but shifted one index to the left
    * @param aPrev previous hidden state
    * @param rnn the RNN model to work with
    * @return tuple of the value of the loss function (cross-entropy) and the last hidden state
    */
  def optimize(xs: List[Int],
               ys: List[Int],
               aPrev: Variable,
               rnn: RnnCell): (Double, Variable) = {
    rnn.zeroGrad()
    val (yHat, a) = rnnForward(xs, aPrev, rnn)
    val loss = rnnLoss(yHat, ys)
    loss.backward()
    rnn.clipGradients(5)
    rnn.step()
    (loss.data.squeeze(), a)
  }

  /**
    * Trains the model and generates dinosaur names
    * @param examples text corpus
    * @param charToIx dictionary that maps a character to an index
    * @param numIterations number of iterations to train the model for
    * @param na number of units of the RNN cell
    * @param numNames number of dinosaur names you want to sample at each iteration
    * @param vocabSize number of unique characters found in the text, size of the vocabulary
    * @param printEvery print stats and samples after 'printEvery' iterations
    */
  def model(examples: List[String],
            charToIx: Map[Char, Int],
            numIterations: Int = 35000,
            na: Int = 50,
            numNames: Int = 7,
            vocabSize: Int = 27,
            printEvery: Int = 1000): Unit = {
    val (nx, ny) = (vocabSize, vocabSize)

    // define the RNN model
    val rnn = RnnCell(na, nx, ny)

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

/**
  *
  * @param wax
  * @param waa
  * @param wya
  * @param ba
  * @param by
  */
case class RnnCell(wax: Variable,
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

  /**
    * Clips the gradients' values between minimum and maximum.
    * @param maxValue everything above this number is set to this number, and everything less than -maxValue is set to -maxValue
    */
  def clipGradients(maxValue: Double): Unit = {
    parameters
      .map(_.grad.get)
      .foreach(v => v.data := ns.clip(v.data, -maxValue, maxValue))
  }

  def step(): Unit = optimizer.step()
}

object RnnCell {
  def apply(na: Int, nx: Int, ny: Int): RnnCell = {
    val wax = Variable(ns.randn(na, nx) * 0.01, name = Some("wax"))
    val waa = Variable(ns.randn(na, na) * 0.01, name = Some("waa"))
    val wya = Variable(ns.randn(ny, na) * 0.01, name = Some("wya"))
    val ba = Variable(ns.zeros(na, 1), name = Some("ba"))
    val by = Variable(ns.zeros(ny, 1), name = Some("by"))
    RnnCell(wax, waa, wya, ba, by)
  }
}

/**
  * Computes the cross-entropy loss
  * @param actuals sequence of yHat variables
  * @param targets sequence of Y indices
  */
case class LossFunction(actuals: Seq[Variable], targets: Seq[Int])
    extends Function {
  /**
    * Computes the cross entropy loss, and wraps it into a variable.
    * The variable can be back propped into, to compute the gradients of the parameters
    * @return the cross entropy loss variable
    */
  override def forward(): Variable = {
    val seqLoss = actuals.zip(targets).foldLeft(0.0) {
      case (loss, (yht, y)) =>
        loss - ns.log(yht.data(y, 0)).squeeze()
    }
    Variable(Tensor(seqLoss), Some(this))
  }

  /**
    * Compute the derivative of the loss, and back prop it into yHat (actuals)
    * @param gradOutput not used
    */
  override def backward(gradOutput: Variable): Unit = {
    actuals.zip(targets).reverse.foreach {
      case (yh, y) =>
        val dy = ns.copy(yh.data)
        dy(y, 0) -= 1
        yh.backward(Variable(dy))
    }
  }
}

/**
  * Sample a sequence of characters according to a sequence of probability distributions output of the RNN
  * @param rnn the network module
  * @param charToIx maps characters to indices
  * @param eolIndex index of the end-of-line character
  */
case class Sampler(rnn: RnnCell, charToIx: Map[Char, Int], eolIndex: Int) {
  val vocabSize: Int = charToIx.size
  val na: Int = rnn.na
  val ixToChar: Map[Int, Char] = charToIx.map(_.swap)

  def generateNextChar(xPrev: Variable,
                       aPrev: Variable): (Variable, Int, Variable) = {
    // Forward propagate x
    val Seq(yHat, aNext) = rnn(xPrev, aPrev)
    // Sample the index of a character within the vocabulary from the probability distribution y
    val nextIdx = ns.choice(ns.arange(vocabSize), yHat.data).squeeze().toInt
    // one hot encoding of the next index
    val xNext = Variable(ns.zerosLike(xPrev.data))
    xNext.data(nextIdx, 0) := 1
    (xNext, nextIdx, aNext)
  }

  /**
    * Loop over time-steps t. At each time-step, sample a character from a probability distribution and append
    * its index to "indices". We'll stop if we reach 50 characters (which should be very unlikely with a well
    * trained model), which helps debugging and prevents entering an infinite loop.
    * @param counter current count
    * @param prevX previous character
    * @param prevA previous activation
    * @param acc accumulator of indices
    * @return acc
    */
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

  def sample(): String = {
    val x0 = Variable(ns.zeros(vocabSize, 1))
    val a0 = Variable(ns.zeros(na, 1))
    val sampledIndices = generate(1, x0, a0, List.empty[Int])
    sampledIndices.map(ixToChar).mkString
  }
}
