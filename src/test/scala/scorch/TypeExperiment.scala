package scorch

import botkop.numsca.Tensor
import org.scalatest.{FlatSpec, Matchers}
import scorch.autograd.Variable

import scala.reflect.runtime.universe
import scala.reflect.runtime.universe._

// case class Parameter(i: Int)

class TypeExperiment extends FlatSpec with Matchers {

  "experiment with type inference" should "" in {

    abstract class A()




  }

}


object TypeExperiment extends App {

  type Parameter = Variable

  abstract class AbstractModule(localParameters: Seq[Parameter] = Nil) {
    def parameters: Unit = {
      this.getClass.getDeclaredFields.toSeq.foreach { f =>

        println(f.getType)

        f setAccessible true
        f.get(this) match {
          case p @ TypeRef(a: universe.Type, sym: universe.Symbol, args: List[universe.Type]) =>
            println(a)
            println(sym)

            Some(p)
          case _            => None
        }
        // ???
      }
    }
  }

  case class MyModule() extends AbstractModule {
    val p = new Parameter(Tensor(3))
    val v = Variable(Tensor(7))
    val a = 10
  }

  val m = MyModule()
  println(m.parameters)

}
