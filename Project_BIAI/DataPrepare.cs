using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


class DataPrepare
{
    public static double[] NormalizeInput(double[] input, double[][] allData)
    {
        const int size = 4; //sepal length, sepal width, petal length, petal width
        double[][] normalizeInput = new double[allData.Length+1][];
        for (int i = 0; i < allData.Length; ++i)
        {
            normalizeInput[i] = new double[size];
            for (int j = 0; j < size; ++j)
                normalizeInput[i][j] = allData[i][j];
        }
        normalizeInput[allData.Length] = new double[size];
        normalizeInput[allData.Length] = input;
        Normalize(normalizeInput, new int[] { 0, 1, 2, 3 }); // first 4 columns only ! 
        return normalizeInput[allData.Length];
    }
    public static void MakeTrainTest(double[][] allData, ref double[][] trainData, ref double[][] testData)
    {
        int rows = allData.Length;
        int columns = allData[0].Length; //7 columns, 4 parameters and 3 indicators of flower
        int trainingRows = (int)(rows * 0.80); //80% trainData
        int testRows = rows - trainingRows; //20% testData

        //new arrays
        trainData = new double[trainingRows][];
        testData = new double[testRows][];

        int[] seq = new int[rows]; // create a random sequence of indexes
        for (int i = 0; i < seq.Length; ++i)
            seq[i] = i; // in cell is number of index for example tab[1] = 1 

        Random random = new Random(0);
        for (int i = 0; i < seq.Length; ++i)
        {
            int r = random.Next(i, seq.Length); //into r random number from ( 0 - how much data)
            int tmp = seq[r]; //saves values from random cell 

            seq[r] = seq[i];
            seq[i] = tmp;
        }

        int sequenceIndex = 0; // index into sequence[]

        //copy 80% of data
        for (int i=0; sequenceIndex < trainingRows; ++sequenceIndex,++i) // first rows to train data 
        {
            trainData[i] = new double[columns];
            int idx = seq[sequenceIndex];
            Array.Copy(allData[idx], trainData[i], columns); //Array.Copy(old, copy, copy.Length);
        }

        //copy 20% of data 
        for (int i=0; sequenceIndex < rows; ++sequenceIndex,++i)
        {
            testData[i] = new double[columns];
            int idx = seq[sequenceIndex];
            Array.Copy(allData[idx], testData[i], columns); //Array.Copy(old, copy, copy.Length);
        }
    }

    public static void Normalize(double[][] dataMatrix, int[] cols)
    {
        // normalize specified cols by computing (x - mean) / sd for each value
        //  Xnew = (x - mean) / sd   -> x - input data, sd = standard deviation, Xnew - normalized data 

        foreach (int col in cols)
        {
            double sum = 0.0;
            for (int i = 0; i < dataMatrix.Length; ++i)
                sum += dataMatrix[i][col]; //for example Sepal Length data (in Column)

            double mean = sum / dataMatrix.Length; //arithmetic average |for example to normalize Sepal Length data

            sum = 0.0;
            for (int i = 0; i < dataMatrix.Length; ++i)
                //error average
                sum += (dataMatrix[i][col] - mean) * (dataMatrix[i][col] - mean); //(x - mean)^2 

            double sd = Math.Sqrt(sum / (dataMatrix.Length - 1)); //standart deviation 

            //normalization formula
            for (int i = 0; i < dataMatrix.Length; ++i)
                dataMatrix[i][col] = (dataMatrix[i][col] - mean) / sd; // Xnew = x - mean / sd 
        }
    }
}

