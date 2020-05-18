import org.bytedeco.javacv.FrameFilter;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;

public class DiabetePrediction {
    public static void main(String[] args) throws Exception {
        MultiLayerNetwork model= ModelSerializer.restoreMultiLayerNetwork(new File("DiabeteModel.zip"));
        String labels[]={"not sick","sick"};
        INDArray inputData= Nd4j.create(new double [][]{
                        {10,125,70,26,115,31.1,0.205,41},/*1*/
                        {7,147,76,0,0,39.4,0.257,43},/*1*/
                {1,97,66,15,140,23.2,0.487,22},/*0*/
                {13,145,82,19,110,22.2,0.245,57},/*0*/
        });

        INDArray output=model.output(inputData);
        int[] classes=output.argMax(1).toIntVector();
        for (int i = 0; i <classes.length ; i++) {
            System.out.println("Classe : "+labels[classes[i]]);
        }

    }
}
