Scorch: neural net framework in Scala
=====================================
Scorch is a minimalistic neural net framework in Scala inspired by PyTorch.


```scala
import scorch.autograd.Variable
import scorch.nn._
import botkop.{numsca => ns}

val numSamples = 128
val numClasses = 10
val nf1 = 40
val nf2 = 20

case class Net() extends Module {
  val fc1 = Linear(nf1, nf2)
  val fc2 = Linear(nf2, numClasses)
  override def subModules(): Seq[Module] = Seq(fc1, fc2)
  override def forward(x: Variable): Variable = fc2(nn.relu(fc1(x)))
}

val n = Net()
val optimizer = SGD(n.parameters(), lr = 0.01)

val target = Variable(ns.randint(numClasses, Array(numSamples, 1)))
val input = Variable(ns.randn(numSamples, nf1))

for (j <- 0 to 1000) {

  optimizer.zeroGrad()

  val output = n(input)
  val loss = nn.softmax(output, target)

  if (j % 100 == 0) {
    val guessed = ns.argmax(output.data, axis = 1)
    val accuracy = ns.sum(target.data == guessed) / numSamples
    println(s"$j: loss: ${loss.data.squeeze()} accuracy: $accuracy")
  }

  loss.backward()
  optimizer.step()
}
```