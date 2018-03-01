package scorch.sandbox.rnn

import botkop.numsca.Tensor
import botkop.{numsca => ns}
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.{FlatSpec, Matchers}
import scorch.TestUtil._
import scorch.autograd._
import scorch.nn._
import scorch._

import scala.language.postfixOps

class WordEmbeddingSpec extends FlatSpec with Matchers {

  Nd4j.setDataType(DataBuffer.Type.DOUBLE)
  ns.rand.setSeed(231)

  "WordEmbedding" should "forward pass" in {
    val (n, t, v, d) = (2, 4, 5, 3)

    val x = Variable(
      ns.array(0, 3, 1, 2, //
          2, 1, 0, 3)
        .reshape(n, t))

    val w = Variable(ns.linspace(0, 1, num = v * d).reshape(v, d))

    val out = WordEmbeddingFunction(x, w).forward()

    val expectedOut = ns
      .array( //
          0, 0.07142857, 0.14285714, //
          0.64285714, 0.71428571, 0.78571429, //
          0.21428571, 0.28571429, 0.35714286, //
          0.42857143, 0.5, 0.57142857, //
          0.42857143, 0.5, 0.57142857, //
          0.21428571, 0.28571429, 0.35714286, //
          0, 0.07142857, 0.14285714, //
          0.64285714, 0.71428571, 0.78571429)
      .reshape(n, t, d)

    val error = relError(out.data, expectedOut)
    println(error)
    assert(error < 1e-7)
  }

  it should "backward pass" in {
    val (n, t, v, d) = (50, 3, 5, 6)
    val x = Variable(ns.randint(v, Array(n, t)))
    val w = Variable(ns.randn(v, d))

    def f(a: Variable): Variable = WordEmbeddingFunction(x, a).forward()
    oneOpGradientCheck(f, w)
  }

  it should "build a simple language model" in {
    // see http://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html#word-embeddings-in-pytorch
    val words = List("hello", "world")
    val wordToIx = words.zipWithIndex.toMap
    val embeds = WordEmbedding(2, 5)
    val lookupTensor = Tensor(wordToIx("hello"))
    val helloEmbed = embeds(Variable(lookupTensor))

    println(helloEmbed.shape)

  }

  it should "build an n-gram language model" in {
    val contextSize = 2
    val embeddingDim = 10
    val testSentence =
      """When forty winters shall besiege thy brow,
        |And dig deep trenches in thy beauty's field,
        |Thy youth's proud livery so gazed on now,
        |Will be a totter'd weed of small worth held:
        |Then being asked, where all thy beauty lies,
        |Where all the treasure of thy lusty days;
        |To say, within thine own deep sunken eyes,
        |Were an all-eating shame, and thriftless praise.
        |How much more praise deserv'd thy beauty's use,
        |If thou couldst answer 'This fair child of mine
        |Shall sum my count, and make my old excuse,'
        |Proving his beauty by succession thine!
        |This were to be new made when thou art old,
        |And see thy blood warm when thou feel'st it cold.
      """.stripMargin

    val tokens = testSentence
      .replaceAll("[\\.',;:\\-]+", "")
      .toLowerCase()
      .split("\\s+")
      .toList
    val trigrams = tokens.sliding(3).toList
    val vocab = tokens.distinct.sorted
    val wordToIx = vocab.zipWithIndex.toMap.mapValues(_.toDouble)

    case class NGramLanguageModeler(vocabSize: Int,
                                    embeddingDim: Int,
                                    contextSize: Int)
        extends Module {

      val embeddings = WordEmbedding(vocabSize, embeddingDim)
      val linear1 = Linear(contextSize * embeddingDim, 128)
      val linear2 = Linear(128, vocabSize)

      override def forward(x: Variable): Variable = {
        val batchSize = x.shape.head
        val embeds = embeddings(x)
          .reshape(batchSize, contextSize * embeddingDim)
        val out1 = relu(linear1(embeds))
        val out2 = linear2(out1)
        out2
      }

      override def subModules: Seq[Module] = Seq(embeddings, linear1, linear2)
    }

    val model = NGramLanguageModeler(vocab.length, embeddingDim, contextSize)
    val optimizer = SGD(model.parameters, lr = 0.1)

    var totalLoss = 0.0

    for (epoch <- 1 to 10) {

      totalLoss = 0

      for (tri <- trigrams) {
        val contextIdxs = tri.init.map(wordToIx)
        val contextVar = Variable(Tensor(contextIdxs: _*))

        model.zeroGrad()

        val output = model(contextVar)
        val target = Variable(Tensor(wordToIx(tri.last)))

        val loss: Variable = softmaxLoss(output, target)
        totalLoss += loss.data.squeeze()

        loss.backward()
        optimizer.step()
      }

      println(s"$epoch: $totalLoss")
    }

    assert(totalLoss < 20)

    val ctxi = Array("dig", "deep").map(wordToIx)
    val ctxv = Variable(Tensor(ctxi))
    val o = model(ctxv)
    val w = vocab(ns.argmax(o.data).squeeze().toInt)
    println(w)

    assert(w == "trenches")
  }

}
