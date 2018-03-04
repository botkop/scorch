package scorch.sandbox.rnn

import botkop.{numsca => ns}
import scorch.autograd.Variable
import scorch.crossEntropyLoss
import scorch.nn.rnn.RnnBase
import scorch.optim.{Adam, Nesterov, Optimizer, SGD}

import scala.annotation.tailrec
import scala.io.Source
import scala.util.Random

class LanguageModel[T](
    corpus: Seq[String],
    tokenize: (String) => Seq[T],
    join: Seq[T] => String,
    eosSymbol: T,
    cellType: String = "gru",
    optimizerType: String = "adam",
    learningRate: Double = 0.001,
    na: Int = 50,
    numIterations: Int = Int.MaxValue,
    maxSentenceSize: Int = 60,
    numSentences: Int = 8,
    printEvery: Int = 100,
) {

  // todo add sorting
  val tokens: Seq[T] =
    corpus.flatMap(tokenize).distinct :+ eosSymbol

  val vocabSize: Int = tokens.length
  val tokenToIdx: Map[T, Int] = tokens.zipWithIndex.toMap

  val bosIndex: Int = -1 // index for beginning of sentence
  val eosIndex: Int = tokenToIdx(eosSymbol) // index for end of sentence
  val (nx, ny) = (vocabSize, vocabSize)

  // define the RNN model
  val rnn = RnnBase(cellType, na, nx, ny)

  val optimizer: Optimizer = optimizerType match {
    case "sgd"      => SGD(rnn.parameters, learningRate)
    case "adam"     => Adam(rnn.parameters, learningRate)
    case "nesterov" => Nesterov(rnn.parameters, learningRate)
    case u          => throw new Error(s"unknown optimizer type $u")
  }

  def optimize(xs: Seq[Int], ys: Seq[Int]): Double = {
    optimizer.zeroGrad()
    val yHat = rnn.forward(xs.map(encode))
    val loss = crossEntropyLoss(yHat, ys)
    loss.backward()
    optimizer.step()
    loss.data.squeeze()
  }

  def run(): Unit = {
    var totalLoss = 0.0
    for (j <- 1 to numIterations) {
      val index = j % corpus.length

      val xs: Seq[Int] = bosIndex +: tokenize(corpus(index)).map(tokenToIdx)
      val ys: Seq[Int] = xs.tail :+ eosIndex

      val loss = optimize(xs, ys)
      totalLoss += loss

      if (j % printEvery == 0) {
        println(s"Iteration: $j, Loss: ${totalLoss / printEvery}")
        (1 to numSentences).foreach(_ => println(sample))
        println()
        totalLoss = 0.0
      }
    }
  }

  def encode(x: Int): Variable = {
    val xt = Variable(ns.zeros(vocabSize, 1))
    if (x != bosIndex)
      xt.data(x, 0) := 1
    xt
  }

  def sample: String = {

    def generateToken(xPrev: Variable,
                      pPrev: Seq[Variable]): (Variable, Int, Seq[Variable]) = {
      // Forward propagate x
      val next = rnn(xPrev +: pPrev: _*)
      val (yHat, pNext) = (next.head, next.tail)
      // Sample the index of a token within the vocabulary from the probability distribution y
      val nextIdx = ns.choice(ns.arange(vocabSize), yHat.data).squeeze().toInt
      // encoding of the next index
      val xNext = encode(nextIdx)
      (xNext, nextIdx, pNext)
    }

    @tailrec
    def generateSequence(t: Int = 1,
                         prevX: Variable = Variable(ns.zeros(vocabSize, 1)),
                         prev: Seq[Variable] = rnn.cell.initialTrackingStates,
                         indices: List[Int] = List.empty): List[Int] =
      if (indices.lastOption.contains(eosIndex)) {
        indices
      } else if (t >= maxSentenceSize) {
        indices :+ eosIndex
      } else {
        val (nextX, nextIdx, nextP) = generateToken(prevX, prev)
        generateSequence(t + 1, nextX, nextP, indices :+ nextIdx)
      }

    val sampledIndices = generateSequence()
    join(sampledIndices.init.map(tokens))
  }

}

object CharModel extends App {

  def tokenize(s: String): Array[Char] = s.toCharArray
  def join(s: Seq[Char]) = s.toString
  val fileName = "src/test/resources/dinos.txt"
  val corpus = Random.shuffle(
    Source
      .fromFile(fileName)
      .getLines()
      .toList)

  val model = new LanguageModel(corpus = corpus,
                                tokenize = tokenize,
                                join = join,
                                eosSymbol = '\n')

  model.run()

}
