This library contains JNI and API for Scala with native code from LibTorch. It uses 
JavaCPP as automatic code generator. 

**Installing:**

Download LibTorch from (https://pytorch.org) for example (https://download.pytorch.org/libtorch/cu90/libtorch-shared-with-deps-latest.zip).   
Add path to the extracted LibTorch in file `build.sbt`:
 
`
JniBuildPlugin.autoImport.torchLibPath in jniBuild := "<path>"
`

Export java home: `` export JAVA_HOME=<path>`` 

Build .so lib: 

>sbt jniBuild
