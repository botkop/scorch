package scorch.examples

import botkop.numsca.Tensor
import botkop.{numsca => ns}
import scorch.autograd.{Function, Variable, softmax, tanh}
import scorch.nn.Optimizer
import scorch.nn.rnn.RecurrentModule

import scala.annotation.tailrec
import scala.io.Source
import scala.util.Random

/**
  * Dinosaurus Island -- Character level language model
  * Scorch implementation of assignment 2 in week 1 of the Coursera course "Recurrent Neural Networks" by deeplearning.ai and Andrew Ng.
  */
object DinosaurIslandCharRnn extends App {
  ns.rand.setSeed(231)
  Random.setSeed(231)

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
    * Instantiates a CrossEntropyLoss object, and applies it
    * @param actuals source for the loss function
    * @param targets targets to compute the loss against
    * @return the loss variable, which can be backpropped into
    */
  def rnnLoss(actuals: Seq[Variable], targets: Seq[Int]): Variable =
    CrossEntropyLoss(actuals, targets).forward()

  /**
    * Performs the forward propagation through the RNN
    * @param xs sequence of input characters to activate
    * @param aPrev the previous hidden state
    * @param rnn the RNN model
    * @param vocabSize vocabulary size
    * @return tuple of the predictions of the RNN over xs, and the hidden state of the last activation
    */
  def rnnForward(xs: List[Int],
                 aPrev: Variable,
                 rnn: RecurrentModule,
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
               rnn: RecurrentModule,
               optimizer: Optimizer): (Double, Variable) = {
    optimizer.zeroGrad()
    val (yHat, a) = rnnForward(xs, aPrev, rnn)
    val loss = rnnLoss(yHat, ys)
    loss.backward()
    optimizer.step()
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

    val optimizer = ClippingSGD(rnn.parameters, maxValue = 5, lr = 0.01)

    val sampler = Sampler(rnn, charToIx, EolIndex, na)

    val aPrev = Variable(ns.zeros(na, 1))

    var totalLoss = 0.0

    for (j <- 1 to numIterations) {
      val index = j % examples.length
      val xs: List[Int] = BolIndex +: examples(index).map(charToIx).toList
      val ys: List[Int] = xs.tail :+ EolIndex

      val (loss, ap) = optimize(xs, ys, aPrev, rnn, optimizer)
      totalLoss += loss
      aPrev.data := ap.data // seems to have little or no effect. Why?

      if (j % printEvery == 0) {
        println(s"Iteration: $j, Loss: ${totalLoss / printEvery}")
        for (_ <- 1 to numNames) {
          print(sampler.sample)
        }
        println()
        totalLoss = 0.0
      }
    }
  }
  model(examples, charToIx, printEvery = 500)
}

/**
  * Module with vanilla RNN activation.
  * Also contains a Gradient Descent optimizer and a method for clipping the gradients to prevent exploding gradients
  * @param wax Weight matrix multiplying the input, variable of shape (na, nx)
  * @param waa Weight matrix multiplying the hidden state, variable of shape (na, na)
  * @param wya Weight matrix relating the hidden-state to the output, variable of shape (ny, na)
  * @param ba Bias, of shape (na, 1)
  * @param by Bias relating the hidden-state to the output, of shape (ny, 1)
  */
case class RnnCell(wax: Variable,
                   waa: Variable,
                   wya: Variable,
                   ba: Variable,
                   by: Variable)
    extends RecurrentModule(Seq(wax, waa, wya, ba, by)) {

  val List(na, nx) = wax.shape
  val List(ny, _) = wya.shape

  override def forward(xs: Seq[Variable]): Seq[Variable] = xs match {
    case Seq(xt, aPrev) =>
      val aNext = tanh(waa.dot(aPrev) + wax.dot(xt) + ba)
      val yt = softmax(wya.dot(aNext) + by)
      Seq(yt, aNext)
  }

}

object RnnCell {

  /**
    * Create an RnnCell from dimensions
    * @param na number of units of the RNN cell
    * @param nx size of the weight matrix multiplying the input
    * @param ny size of the weight matrix relating the hidden-state to the output
    * @return a vanilla Rnn model
    */
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
case class CrossEntropyLoss(actuals: Seq[Variable], targets: Seq[Int])
    extends Function {

  /**
    * Computes the cross entropy loss, and wraps it in a variable.
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
    * Compute the loss of each generated character, and back prop from last to first
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
  * Stochastic Gradient Descent with clipping
  * @param parameters list of parameters to optimize
  * @param maxValue  gradients above this number are set to this number, and everything less than -maxValue is set to -maxValue
  * @param lr learning rate
  */
case class ClippingSGD(parameters: Seq[Variable], maxValue: Double, lr: Double)
    extends Optimizer(parameters) {
  override def step(): Unit = {
    parameters.foreach { p =>
      p.data -= ns.clip(p.grad.get.data, -maxValue, maxValue) * lr
    }
  }
}

/**
  * Sample a sequence of characters according to a sequence of probability distributions output of the RNN
  * @param rnn the network module
  * @param charToIx maps characters to indices
  * @param eolIndex index of the end-of-line character
  * @param na number of units of the RNN cell
  */
case class Sampler(rnn: RecurrentModule,
                   charToIx: Map[Char, Int],
                   eolIndex: Int,
                   na: Int) {
  val vocabSize: Int = charToIx.size
  val ixToChar: Map[Int, Char] = charToIx.map(_.swap)

  /**
    * Reuse the previously generated character and hidden state to generate the next character
    * @param xPrev the previous character
    * @param aPrev the previous hidden state
    * @return tuple of (next character, index of the next character, the next hidden state)
    */
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
    * Recurse over time-steps t. At each time-step, sample a character from a probability distribution and append
    * its index to "indices". We'll stop if we reach 50 characters (which should be very unlikely with a well
    * trained model), which helps debugging and prevents entering an infinite loop.
    * @param counter current count
    * @param prevX previous character
    * @param prevA previous activation
    * @param indices accumulator of indices
    * @return acc
    */
  @tailrec
  final def generate(counter: Int,
                     prevX: Variable,
                     prevA: Variable,
                     indices: List[Int]): List[Int] =
    if (indices.lastOption.contains(eolIndex)) {
      indices
    } else if (counter >= 50) {
      indices :+ eolIndex
    } else {
      val (nextX, nextIdx, nextA) = generateNextChar(prevX, prevA)
      generate(counter + 1, nextX, nextA, indices :+ nextIdx)
    }

  def sample: String = {
    val x0 = Variable(ns.zeros(vocabSize, 1))
    val a0 = Variable(ns.zeros(na, 1))
    val sampledIndices = generate(1, x0, a0, List.empty[Int])
    sampledIndices.map(ixToChar).mkString
  }
}
