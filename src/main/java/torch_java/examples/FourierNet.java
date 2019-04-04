package torch_java.examples;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;
import torch_java.api.nn.Module;

@Platform(
        include = {
                "torch/all.h",
                "models/FourierNet.h",
                "<iostream>",
                "<vector>",
                "<map>"
        })

@NoOffset public class FourierNet extends Module {


    public FourierNet(int size) {
        super();
        allocate(size);
    }

    public native void allocate(int size);

    // std::pair<float, std::vector<float> > train(std::vector<float> data, int steps, std::vector<float> weights)

    private native @StdVector FloatPointer train (@StdVector FloatPointer data, int steps, @StdVector FloatPointer weights);

    public float[] train(float[] data, int steps, float[] weights) {
        FloatPointer res = train(new FloatPointer(data), steps, new FloatPointer(weights));
        float[] res_array = new float[data.length];
        for(int i = 0; i < res_array.length; i++) {
            res_array[i] = res.get(i);
        }
        return res_array;
    }

    public float loss(float[] y1, float[] y2, float[] weights) {
        float res = 0;
        int n = y1.length;
        for(int i = 0; i < n; i++) {
            float dy = y1[i] - y2[i];
            res += dy * dy * weights[i];
        }
        return res;
    }

    public static void main(String[] args) throws InterruptedException {

        FourierNet net = new FourierNet(20);
        float[] pred = net.train(new float[]{1, 2, 3, 3, 4, 5, 6, 7, 6, 4, 3, 2, 4, 4, 5}, 500, new float[]{1,1,1,1,1,1,1,1,1,1,1,1,1,1,1});
        float loss = net.loss(pred, new float[]{1, 2, 3, 3, 4, 5, 6, 7, 6, 4, 3, 2, 4, 4, 5}, new float[]{1,1,1,1,1,1,1,1,1,1,1,1,1,1,1});
        System.out.println("loss = " + loss);
    }

}
