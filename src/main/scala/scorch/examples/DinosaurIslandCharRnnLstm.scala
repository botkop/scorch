package scorch.examples

import botkop.numsca.Tensor
import botkop.{numsca => ns}
import scorch.autograd._
import scorch.nn.rnn.RecurrentModule
import scorch.nn.Optimizer

import scala.annotation.tailrec
import scala.io.Source
import scala.util.Random

/**
  * Dinosaurus Island -- Character level language model
  * Assignment 2 in week 1 of the Coursera course "Recurrent Neural Networks" by deeplearning.ai and Andrew Ng.
  * Implementation in Scala with Scorch.
  * Scorch lives here: https://github.com/botkop/scorch
  *
  * We will build a character level language model to generate new dinosaur names.
  * The algorithm will learn the different name patterns, and randomly generate new names.
  * The dataset to train on is in "src/test/resources/dinos.txt".
  * Both vanilla RNN and LSTM are provided.
  */
object DinosaurIslandCharRnnLstm extends App {
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
  val EosIndex = charToIx('\n') // index for end of sentence
  val BosIndex = -1 // index for beginning of sentence

  model("rnn", examples, charToIx, na = 10, printEvery = 100)

  /**
    * Trains the model and generates dinosaur names
    * @param cellType "rnn" or "lstm"
    * @param examples text corpus
    * @param charToIx dictionary that maps a character to an index
    * @param numIterations number of iterations to train the model for
    * @param na number of units of the RNN cell
    * @param numNames number of dinosaur names you want to sample at each 'printEvery' iteration
    * @param vocabSize number of unique characters found in the text, size of the vocabulary
    * @param printEvery print stats and samples after 'printEvery' iterations
    */
  def model(cellType: String,
            examples: List[String],
            charToIx: Map[Char, Int],
            numIterations: Int = 35000,
            na: Int = 50,
            numNames: Int = 7,
            vocabSize: Int = 27,
            printEvery: Int = 1000): Unit = {
    val (nx, ny) = (vocabSize, vocabSize)

    // define the RNN model
    val rnn = cellType match {
      case "rnn"  => RnnCell(na, nx, ny)
      case "lstm" => LstmCell(na, nx, ny)
      case u      => throw new Error(s"unknown cell type $u")
    }

    val optimizer = ClippingSGD(rnn.parameters, maxValue = 5, lr = 0.05)

    val sampler = Sampler(rnn, charToIx, EosIndex, na)

    var totalLoss = 0.0

    for (j <- 1 to numIterations) {
      val index = j % examples.length
      val xs: List[Int] = BosIndex +: examples(index).map(charToIx).toList
      val ys: List[Int] = xs.tail :+ EosIndex

      val pPrev = rnn.initialTrackingStates

      val loss = optimize(xs, ys, pPrev, rnn, optimizer)
      totalLoss += loss

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
    * @param previous the previous hidden state(s)
    * @param rnn the RNN model
    * @param vocabSize vocabulary size
    * @return tuple of the predictions of the RNN over xs, and the hidden state of the last activation
    */
  def rnnForward(xs: List[Int],
                 previous: Seq[Variable],
                 rnn: RecurrentModule,
                 vocabSize: Int = 27): List[Variable] =
    xs.foldLeft(List.empty[Variable], previous) {
        case ((yhs, p0), x) =>
          // one hot encoding of next x
          val xt = Variable(ns.zeros(vocabSize, 1))
          if (x != BosIndex)
            xt.data(x, 0) := 1

          val parameters = xt +: p0
          val next = rnn(parameters: _*)
          val (yht, p1) = (next.head, next.tail)
          (yhs :+ yht, p1)
      }
      ._1

  /**
    * Execute one step of the optimization to train the model
    * @param xs list of integers, where each integer is a number that maps to a character in the vocabulary
    * @param ys list of integers, exactly the same as xs but shifted one index to the left
    * @param previous previous hidden state(s)
    * @param rnn the RNN model to work with
    * @return tuple of the value of the loss function (cross-entropy) and the last hidden state
    */
  def optimize(xs: List[Int],
               ys: List[Int],
               previous: Seq[Variable],
               rnn: RecurrentModule,
               optimizer: Optimizer): Double = {
    optimizer.zeroGrad()
    val yHat = rnnForward(xs, previous, rnn)
    val loss = rnnLoss(yHat, ys)
    loss.backward()
    optimizer.step()
    loss.data.squeeze()
  }

  /**
    * Extension of RecurrentModule to allow storing the number of previous states
    * and a method for generating the initial states
    * @param vs local parameters of the module
    */
  abstract class BaseRnnCell(vs: Seq[Variable]) extends RecurrentModule(vs) {

    /**
      * Number of units in the cell
      */
    def na: Int

    /**
      * Number of states to keep track of.
      * For example, in an LSTM you keep track of the hidden state and the cell state.
      * In this case this would be = 2
      * For an RNN, it would be 1, because you only keep track of the hidden state.
      */
    def numTrackingStates: Int

    /**
      * Provide a sequence of states to start an iteration with
      */
    def initialTrackingStates: Seq[Variable] =
      Seq.fill(numTrackingStates)(Variable(ns.zeros(na, 1)))
  }

  /**
    * Implements a single forward step of the LSTM-cell
    * @param wf Weight matrix of the forget gate, numpy array of shape (na, na + nx)
    * @param bf Bias of the forget gate, numpy array of shape (na, 1)
    * @param wi Weight matrix of the update gate, numpy array of shape (na, na + nx)
    * @param bi Bias of the update gate, numpy array of shape (na, 1)
    * @param wc Weight matrix of the first "tanh", numpy array of shape (na, na + nx)
    * @param bc Bias of the first "tanh", numpy array of shape (na, 1)
    * @param wo Weight matrix of the output gate, numpy array of shape (na, na + nx)
    * @param bo Bias of the output gate, numpy array of shape (na, 1)
    * @param wy Weight matrix relating the hidden-state to the output, numpy array of shape (ny, na)
    * @param by Bias relating the hidden-state to the output, numpy array of shape (ny, 1)
    */
  case class LstmCell(
      wf: Variable,
      bf: Variable,
      wi: Variable,
      bi: Variable,
      wc: Variable,
      bc: Variable,
      wo: Variable,
      bo: Variable,
      wy: Variable,
      by: Variable
  ) extends BaseRnnCell(Seq(wf, bf, wi, bi, wc, bc, wo, bo, wy, by)) {
    val List(ny, na) = wy.shape
    override val numTrackingStates = 2

    /**
      * Lstm cell forward pass
      * @param xs sequence of Variables:
      *           - xt: your input data at timestep "t", of shape (nx, m).
      *           - aPrev: Hidden state at timestep "t-1", of shape (na, m)
      *           - cPrev: Memory state at timestep "t-1", numpy array of shape (na, m)
      * @return sequence of Variables:
      *         - aNext: next hidden state, of shape (na, m)
      *         - cNext: next memory state, of shape (na, m)
      *         - ytHat: prediction at timestep "t", of shape (ny, m)
      */
    override def forward(xs: Seq[Variable]): Seq[Variable] = xs match {
      case Seq(xt, aPrev, cPrev) =>
        // val concat = scorch.cat(aPrev, xt) // not clear why this is causing endless recursion

        val concat = Variable(ns.concatenate(Seq(aPrev.data, xt.data)))

        // Forget gate
        val ft = sigmoid(wf.dot(concat) + bf)
        // Update gate
        val it = sigmoid(wi.dot(concat) + bi)
        val cct = tanh(wc.dot(concat) + bc)
        val cNext = ft * cPrev + it * cct
        // Output gate
        val ot = sigmoid(wo.dot(concat) + bo)
        val aNext = ot * tanh(cNext)
        val ytHat = softmax(wy.dot(aNext) + by)
        Seq(ytHat, aNext, cNext)
    }
  }

  object LstmCell {

    /**
      * Create an LstmCell from dimensions
      * @param na number of units of the LstmCell
      * @param nx size of the weight matrix multiplying the input
      * @param ny size of the weight matrix relating the hidden-state to the output
      * @return initialized Lstm cell
      */
    def apply(na: Int, nx: Int, ny: Int): LstmCell = {
      val wf = Variable(ns.randn(na, na + nx) * 0.01, name = Some("wf"))
      val bf = Variable(ns.zeros(na, 1), name = Some("bf"))
      val wi = Variable(ns.randn(na, na + nx) * 0.01, name = Some("wi"))
      val bi = Variable(ns.zeros(na, 1), name = Some("bi"))
      val wc = Variable(ns.randn(na, na + nx) * 0.01, name = Some("wc"))
      val bc = Variable(ns.zeros(na, 1), name = Some("bc"))
      val wo = Variable(ns.randn(na, na + nx) * 0.01, name = Some("bo"))
      val bo = Variable(ns.zeros(na, 1), name = Some("by"))
      val wy = Variable(ns.randn(ny, na) * 0.01, name = Some("wy"))
      val by = Variable(ns.zeros(ny, 1), name = Some("ba"))
      LstmCell(wf, bf, wi, bi, wc, bc, wo, bo, wy, by)
    }
  }

  /**
    * Module with vanilla RNN activation.
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
      extends BaseRnnCell(Seq(wax, waa, wya, ba, by)) {

    val List(na, nx) = wax.shape
    val List(ny, _) = wya.shape
    override val numTrackingStates: Int = 1

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
    * @param targets sequence of Y indices (ground truth)
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
    * Sample a sequence of characters according to a sequence of probability distributions output of the RNN
    * @param rnn the network module
    * @param charToIx maps characters to indices
    * @param eolIndex index of the end-of-line character
    * @param na number of units of the RNN cell
    */
  case class Sampler(rnn: BaseRnnCell,
                     charToIx: Map[Char, Int],
                     eolIndex: Int,
                     na: Int) {
    val vocabSize: Int = charToIx.size
    val ixToChar: Map[Int, Char] = charToIx.map(_.swap)

    def generateNextChar(
        xPrev: Variable,
        pPrev: Seq[Variable]): (Variable, Int, Seq[Variable]) = {
      // Forward propagate x
      val next = rnn(xPrev +: pPrev: _*)
      val (yHat, pNext) = (next.head, next.tail)
      // Sample the index of a character within the vocabulary from the probability distribution y
      val nextIdx = ns.choice(ns.arange(vocabSize), yHat.data).squeeze().toInt
      // one hot encoding of the next index
      val xNext = Variable(ns.zerosLike(xPrev.data))
      xNext.data(nextIdx, 0) := 1
      (xNext, nextIdx, pNext)
    }

    @tailrec
    final def generate(t: Int,
                       prevX: Variable,
                       prev: Seq[Variable],
                       indices: List[Int] = List.empty): List[Int] =
      if (indices.lastOption.contains(eolIndex)) {
        indices
      } else if (t >= 50) {
        indices :+ eolIndex
      } else {
        val (nextX, nextIdx, nextP) = generateNextChar(prevX, prev)
        generate(t + 1, nextX, nextP, indices :+ nextIdx)
      }

    def sample: String = {
      val x0 = Variable(ns.zeros(vocabSize, 1))
      val p0 = rnn.initialTrackingStates
      val sampledIndices = generate(1, x0, p0)
      sampledIndices.map(ixToChar).mkString
    }
  }

  /**
    * Stochastic Gradient Descent with clipping to prevent exploding gradients
    * @param parameters list of parameters to optimize
    * @param maxValue  gradients above this number are set to this number, and everything less than -maxValue is set to -maxValue
    * @param lr learning rate
    */
  case class ClippingSGD(parameters: Seq[Variable],
                         lr: Double,
                         maxValue: Double)
      extends Optimizer(parameters) {
    override def step(): Unit =
      parameters.foreach { p =>
        p.data -= ns.clip(p.grad.get.data, -maxValue, maxValue) * lr
      }
  }
}
