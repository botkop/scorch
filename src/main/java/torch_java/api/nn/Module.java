package torch_java.api.nn;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Platform(
        include = {
                "torch/all.h",
                "torch/nn/module.h"
        })

@Namespace("torch::nn") @NoOffset public class Module extends Pointer {

    static {
        String workingDir = System.getProperty("user.dir");
        System.load( workingDir + "/sosca/libjava_torch_lib.so");
    }

    public Module() {
        super((Pointer) null);
        allocate();
    }

    public native void allocate();
}