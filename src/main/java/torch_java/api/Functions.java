package torch_java.api;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Platform(
        include = {
                "ATen/ATen.h",
                "torch/csrc/autograd/generated/variable_factories.h",
                "ATen/core/TensorOptions.h",
                "c10/util/ArrayRef.h",
                "helper.h",
                "models/FourierNet.h"
        })


@Namespace("at") @NoOffset public  class Functions {

    static {
        System.loadLibrary("libjava_torch_lib");
    }


    @Opaque public static class TensorOptions extends Pointer {
        /** Empty constructor. Calls {@code super((Pointer)null)}. */
        public TensorOptions() { super((Pointer)null); allocate(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public TensorOptions(Pointer p) { super(p); }
        public native void allocate();
    }

    @Opaque public static class Type extends Pointer {
        public Type() { super((Pointer)null);  }
        public Type(Pointer p) { super(p); }
    }


    @Opaque public static class IntList extends Pointer {
        public IntList() { super((Pointer)null);  }
        public IntList(Pointer p) { super(p); }
        public IntList(long[] data) { super((Pointer)null);  allocate(data, data.length);}
        public native void allocate(@Cast("int64_t*") long[] data, @Cast("size_t") int length);
        public native @Cast("const int64_t*") long[] data();
    }



    // public static native @ByVal Tensor make_ones(@Cast("long *") long[] dims, @Cast("size_t") long size, int dtype);

    public static native @StdVector FloatPointer train (@StdVector FloatPointer data, int steps, @StdVector FloatPointer weights);

}
