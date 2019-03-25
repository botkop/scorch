package org.bytedeco.javacpp.presets;

import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.javacpp.tools.*;

@Properties(target="org.bytedeco.javacpp.java_torch", value={@Platform(include="/home/nazar/libtorch/include/ATen/ATen.h", link="java_torch@.1"),
        @Platform(value="linux", link="so", preload="java_torch1")})
public class torch implements InfoMapper {
    public void map(InfoMap infoMap) {


    }
}