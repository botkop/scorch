package torch_scala

import org.bytedeco.javacpp.annotation.{Cast, Namespace, Platform}
import org.bytedeco.javacpp.tools.{InfoMap, InfoMapper}
import org.bytedeco.javacpp._
import org.bytedeco.javacpp.annotation._
import org.bytedeco.javacpp.tools._


//@Properties(target = "torch_native_lib1234",
//            value = Array(new Platform(include = Array("torch/all.h")))
//)
//class NativeLibraryConfig extends InfoMapper {
//  def map(infoMap: InfoMap): Unit = {
//    //infoMap.put(new Info("data<long>").javaNames("data_int"))
//  }
//}


trait NativeLoader {
  val workingDir: String = System.getProperty("user.dir")
  System.load(workingDir + "/src/native/libjava_torch_lib0.so")

}


@Platform(include = Array("torch/all.h"))
@Namespace("torch::cuda") object Torch {

  @native @Cast(Array("size_t")) def device_count(): Int

  /// Returns true if at least one CUDA device is available.
  @native def is_available(): Boolean

  /// Returns true if CUDA is available, and CuDNN is available.
  @native def cudnn_is_available(): Boolean

}
