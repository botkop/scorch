package torch_scala.api.nn

import org.bytedeco.javacpp._
import org.bytedeco.javacpp.annotation._


@Platform(include = Array(
  "torch/all.h",
  "torch/nn/module.h"
))
@Namespace("torch::nn") @NoOffset class Module() extends Pointer {


  val workingDir = System.getProperty("user.dir")
  System.load(workingDir + "/src/native/libjava_torch_lib0.so")


  allocate()

  @native def allocate(): Unit
}