package scorch.sandbox.rnn

import botkop.{numsca => ns}
import scorch.autograd.Variable
import scorch.crossEntropyLoss
import scorch.nn.rnn.RnnBase
import scorch.optim.{Adam, Nesterov, Optimizer, SGD}

import scala.annotation.tailrec

class LanguageModel[T](
    corpus: Seq[String],
    tokenize: (String) => Seq[T],
    join: Seq[T] => String,
    encoder: (Int) => Variable,
    eosSymbol: T,
    cellType: String = "gru",
    optimizerType: String = "adam",
    learningRate: Double = 0.06,
    na: Int = 50,
    numIterations: Int = Int.MaxValue,
    maxSentenceSize: Int = 60,
    numSentences: Int = 8,
    printEvery: Int = 100,
) {

  val tokens: Seq[T] =
    tokenize(corpus.mkString(" ")).distinct.sorted :+ eosSymbol

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
    val yHat = rnn.forward(xs.map(encoder))
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
        (1 to numSentences).foreach(_ => println(sample()))
        println()
        totalLoss = 0.0
      }
    }
  }

  def sample(): String = {

    def generateNextToken(
        xPrev: Variable,
        pPrev: Seq[Variable]): (Variable, Int, Seq[Variable]) = {
      // Forward propagate x
      val next = rnn(xPrev +: pPrev: _*)
      val (yHat, pNext) = (next.head, next.tail)
      // Sample the index of a token within the vocabulary from the probability distribution y
      val nextIdx = ns.choice(ns.arange(vocabSize), yHat.data).squeeze().toInt
      // encoding of the next index
      val xNext = encoder(nextIdx)
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
        val (nextX, nextIdx, nextP) = generateNextToken(prevX, prev)
        generateSequence(t + 1, nextX, nextP, indices :+ nextIdx)
      }

    val sampledIndices = generateSequence()
    join(sampledIndices.init.map(tokens))
  }

}
