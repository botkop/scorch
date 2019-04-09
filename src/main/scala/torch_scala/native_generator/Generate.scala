package torch_scala.native_generator

import generate.Builder
import org.bytedeco.javacpp.Loader
import org.bytedeco.javacpp.tools.{Generator, Logger}
import torch_scala.api.nn.Module
import torch_scala.examples.FourierNet

object Generate extends App {

//  val gen = new Generator(Logger.create(classOf[Generator]), Loader.loadProperties)

  //  val res = gen.generate("/home/nazar/java_torch_2/src/native/java_torch_lib.cpp", "/home/nazar/java_torch_2/src/native/java_torch_lib.h", "", "", "",
  //    classOf[FourierNet],
  //    classOf[Module]
  //  )

  val gen = new Builder()
  gen.outputDirectory("src/native")
  gen.classesOrPackages("torch_scala.api.nn.Module", "torch_scala.examples.FourierNet")
  gen.build()

//  println(res)

}
