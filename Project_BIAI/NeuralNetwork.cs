using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


public class NeuralNetwork
{
    private int input;
    private int hidden;
    private int output;

    //Values - > neurons
    private double[] inputs; //values of input neurons
    private double[] outputs;//values of output neurons
    private double[] hiddens;//values of hidden neurons

    //Weights
    private double[][] weightsIH; // input-hidden weights
    private double[][] weightsHO; // hidden-output
    private double[] hiddenBiases; //hidden - biases
    private double[] outputBiases;

    //BACK-PROPAGATION specific arrays 
    private double[] outputGradient; // output gradients for back-propagation
    private double[] hiddenGradient; // hidden gradients

    // back-prop previous weights -> momentum specific arrays 
    private double[][] ihPrevWeightsDelta;  // for momentum with back-propagation
    private double[] hiddenPrevBiasesDelta;
    private double[][] hoPrevWeightsDelta;
    private double[] outputPrevBiasesDelta;
    private static Random r;

    public NeuralNetwork(int numInput, int numHidden, int numOutput)
    {
        this.input = numInput; //number of: input hidden and output neurons,
        this.hidden = numHidden;
        this.output = numOutput;

        this.inputs = new double[numInput]; //values
        this.outputs = new double[numOutput];
        this.hiddens = new double[numHidden];

        this.weightsIH = createMatrix(numInput, numHidden); //weights for neural network
        this.weightsHO = createMatrix(numHidden, numOutput);
        this.outputBiases = new double[numOutput];
        this.hiddenBiases = new double[numHidden];

        // back-prop
        this.hiddenGradient = new double[numHidden];
        this.outputGradient = new double[numOutput];

        this.ihPrevWeightsDelta = createMatrix(numInput, numHidden);
        this.hiddenPrevBiasesDelta = new double[numHidden];
        this.hoPrevWeightsDelta = createMatrix(numHidden, numOutput);
        this.outputPrevBiasesDelta = new double[numOutput];
        r = new Random(0); //shuffle,initWeights
    }

    private static double[][] createMatrix(int rows, int cols)
    {
        //Create matrix
        double[][] matrix = new double[rows][];
        for (int r = 0; r < matrix.Length; ++r)
            matrix[r] = new double[cols];
        return matrix;
    }


    private static void shuffle(int[] indexs)
    {
        for (int i = 0; i < indexs.Length; ++i)
        {
            int r = NeuralNetwork.r.Next(i, indexs.Length);
            int tmp = indexs[r];
            indexs[r] = indexs[i];
            indexs[i] = tmp;
        }
    }

    public void setWeights(double[] weights)
    {
        // copy all weights and all biases in weights[] array 
        //to i-h weights, i-h biases, h-o weights, h-o biases

        int helper = 0;

        for (int i = 0; i < input; ++i)
            for (int j = 0; j < hidden; ++j)
                weightsIH[i][j] = weights[helper++];
        for (int i = 0; i < hidden; ++i)
            hiddenBiases[i] = weights[helper++];
        for (int i = 0; i < hidden; ++i)
            for (int j = 0; j < output; ++j)
                weightsHO[i][j] = weights[helper++];
        for (int i = 0; i < output; ++i)
            outputBiases[i] = weights[helper++];
    }

    public void InitializeWeights()
    {
        // initialize weights and biases
        int numberOfWeights = (input * hidden) + (hidden * output) + hidden + output;
        double[] initWeights = new double[numberOfWeights];
        double lo = -0.01;
        double hi = 0.01;
        for (int i = 0; i < initWeights.Length; ++i)
        {
            initWeights[i] = (hi - lo) * r.NextDouble() + lo; //init to small random values
        }
        this.setWeights(initWeights);
    }

    public double[] getWeights()
    {
        // return weights, after training
        int numberOfWeights = (input * hidden) + (hidden * output) + hidden + output;
        double[] result = new double[numberOfWeights];
        int helper = 0;
        for (int i = 0; i < weightsIH.Length; ++i)
            for (int j = 0; j < weightsIH[0].Length; ++j)
                result[helper++] = weightsIH[i][j];
        for (int i = 0; i < hiddenBiases.Length; ++i)
            result[helper++] = hiddenBiases[i];
        for (int i = 0; i < weightsHO.Length; ++i)
            for (int j = 0; j < weightsHO[0].Length; ++j)
                result[helper++] = weightsHO[i][j];
        for (int i = 0; i < outputBiases.Length; ++i)
            result[helper++] = outputBiases[i];
        return result;
    }

    // Performing feedforward algorithm
    public double[] forward(double[] values)
    {
        double[] hiddenSums = new double[hidden]; // sums of hidden nodes 
        double[] outputSums = new double[output]; // sums of output nodes

        // copy values to inputs
        for (int i = 0; i < values.Length; ++i)
            this.inputs[i] = values[i];

        for (int j = 0; j < hidden; ++j)  // i-h sum of weights * inputs
            for (int i = 0; i < input; ++i)
                hiddenSums[j] += this.inputs[i] * this.weightsIH[i][j];

        for (int i = 0; i < hidden; ++i)  //add biases to input-to-hidden sums
            hiddenSums[i] += this.hiddenBiases[i];

        for (int i = 0; i < hidden; ++i)   // ACTIVATION
            this.hiddens[i] = hyperTang(hiddenSums[i]);

        for (int j = 0; j < output; ++j)   //h-o sum of weights * hOutputs
            for (int i = 0; i < hidden; ++i)
                outputSums[j] += hiddens[i] * weightsHO[i][j];

        for (int i = 0; i < output; ++i)  // add biases to input-to-hidden sums
            outputSums[i] += outputBiases[i];

        double[] softOutput = softmax(outputSums); // softmax activation 
        Array.Copy(softOutput, outputs, softOutput.Length);

        double[] result = new double[output];
        Array.Copy(this.outputs, result, result.Length);
        return result;
    }

    private static double hyperTang(double t)
    {
        if (t < -20.0) return -1.0;
        else if (t > 20.0) return 1.0;
        else return Math.Tanh(t);
    }

    private static double[] softmax(double[] outputSums)
    {
        double max = outputSums[0];
        for (int i = 0; i < outputSums.Length; ++i)
            if (outputSums[i] > max) max = outputSums[i];

        // determine scaling factor --> sum of exp(each val - max)
        double scale = 0.0;
        for (int i = 0; i < outputSums.Length; ++i)
            scale += Math.Exp(outputSums[i] - max);

        double[] result = new double[outputSums.Length];
        for (int i = 0; i < outputSums.Length; ++i)
            result[i] = Math.Exp(outputSums[i] - max) / scale;

        return result;
    }


    private void backPropagation(double[] targetValues, double learnRate, double momentum, double weightDecay)
    {

        //output gradients
        for (int i = 0; i < outputGradient.Length; ++i)
        {
            //2 - derivative of softmax = (1 - y) * y
            double derivative = (1 - outputs[i]) * outputs[i];
            // (1-y)(y) derivative
            outputGradient[i] = derivative * (targetValues[i] - outputs[i]); //formula 5 error signal - (targetValues[i] - outputs[i]), formula - 6 error signal * output value
        }

        //hidden gradients
        for (int i = 0; i < hiddenGradient.Length; ++i)
        {   //for hidden layers node using tanh
            // derivative of tanh = (1 - y) * (1 + y)
            double derivative = (1 - hiddens[i]) * (1 + hiddens[i]); //3 - derivative (1-y) * (1+y)
            double sum = 0.0;
            for (int j = 0; j < output; ++j)
            {
                double x = outputGradient[j] * weightsHO[i][j]; //sum need in formula (7) hidden node
                sum += x;
            }
            hiddenGradient[i] = derivative * sum; // 7 local error gradient signal - hiddenGradient
        }

        //update hidden weights - 8
        for (int i = 0; i < weightsIH.Length; ++i)
        {
            for (int j = 0; j < weightsIH[0].Length; ++j)
            {
                double delta = learnRate * hiddenGradient[j] * inputs[i]; // 1 - (hiddenGradient * inputs[i] (xi)), 8 = 1 * learn rate
                weightsIH[i][j] += delta;

                //adding momentum to previous delta
                weightsIH[i][j] += momentum * ihPrevWeightsDelta[i][j];
                weightsIH[i][j] -= (weightDecay * weightsIH[i][j]); // weight decay
                ihPrevWeightsDelta[i][j] = delta; // saving !!!!
            }
        }

        //update hidden biases - 1
        for (int i = 0; i < hiddenBiases.Length; ++i)
        {
            double delta = learnRate * hiddenGradient[i] * 1.0; //1.0 for bias 
            hiddenBiases[i] += delta;
            hiddenBiases[i] += momentum * hiddenPrevBiasesDelta[i]; // momentum
            hiddenBiases[i] -= (weightDecay * hiddenBiases[i]); // weight decay
            hiddenPrevBiasesDelta[i] = delta; //saving !!!
        }

        //update hidden-output weights
        for (int i = 0; i < weightsHO.Length; ++i)
        {
            for (int j = 0; j < weightsHO[0].Length; ++j)
            {
                double delta = learnRate * outputGradient[j] * hiddens[i];
                weightsHO[i][j] += delta;
                weightsHO[i][j] += momentum * hoPrevWeightsDelta[i][j]; // momentum
                weightsHO[i][j] -= (weightDecay * weightsHO[i][j]); // weight decay
                hoPrevWeightsDelta[i][j] = delta; //saving !!!
            }
        }

        //update output biases
        for (int i = 0; i < outputBiases.Length; ++i)
        {
            double delta = learnRate * outputGradient[i] * 1.0; //8 - delta for all weights and biases 
            outputBiases[i] += delta; //update for all weights and biases 
            outputBiases[i] += momentum * outputPrevBiasesDelta[i]; // momentum
            outputBiases[i] -= (weightDecay * outputBiases[i]); // weight decay
            outputPrevBiasesDelta[i] = delta; // save
        }
    }
        
    public void training(double[][] trainData, int epochs, double learnRate, double momentum, double weightDecay)
    {
        int epoch = 0;
        double[] inputValues = new double[input]; // inputs
        double[] targetValues = new double[output]; // target values

        int[] indexs = new int[trainData.Length];
        for (int i = 0; i < indexs.Length; ++i)
            indexs[i] = i;
        
        while (epoch < epochs)
        {
            /* mean squared error (pl:błąd średniokwadratowy) < 0.020 - stopping condition */
            double helper = meanSquaredError(trainData);
            if (helper < 0.020)
            {
                break;
            }

            shuffle(indexs);
            for (int i = 0; i < trainData.Length; ++i)
            {
                int j = indexs[i];
                Array.Copy(trainData[j], inputValues, input);
                Array.Copy(trainData[j], input, targetValues, 0, output);
                forward(inputValues); //store
                backPropagation(targetValues, learnRate, momentum, weightDecay); // find better weights using backpropagation
            }
            ++epoch;
        }
    }

    private double meanSquaredError(double[][] data) //trainData 
    {
        double sumSquaredError = 0.0;
        double[] inputValues = new double[input];
        double[] targetValues = new double[output];

        for (int i = 0; i < data.Length; ++i)
        {
            Array.Copy(data[i], inputValues, input);
            Array.Copy(data[i], input, targetValues, 0, output); // get target values
            double[] myOutputValues = this.forward(inputValues); // compute output and use current weights
            for (int j = 0; j < output; ++j)
            {
                double error = targetValues[j] - myOutputValues[j];
                sumSquaredError += error * error;
            }
        }

        return sumSquaredError / data.Length;
    }

    private static int maxIndex(double[] vector) // helper for Accuracy()
    {
        // search an index with max value
        int result = 0;
        double biggestVal = vector[0];
        for (int i = 0; i < vector.Length; ++i)
        {
            if (vector[i] > biggestVal)
            {
                biggestVal = vector[i]; result = i;
            }
        }
        return result;
    }

    public double accuracy(double[][] data) //testData % % %
    {
        double[] inputValues = new double[input]; // inputs
        double[] targetValues = new double[output]; // targets
        double[] myOutputValues; // outputs

        int correct = 0;
        int wrong = 0;
        for (int i = 0; i < data.Length; ++i)
        {
            Array.Copy(data[i], inputValues, input); // data into inputValues 
            Array.Copy(data[i], input, targetValues, 0, output); //and target Values 

            // FEEDFORWARD
            myOutputValues = this.forward(inputValues);
            int placeOf1 = maxIndex(myOutputValues); // where is 1 ? ? ? ?

            if (targetValues[placeOf1] == 1.0) // if 1 is in good place 
                ++correct;
            else
                ++wrong; //if not
        }
        int all = correct + wrong;
        return (correct * 1.0) / (all);
    }
}
