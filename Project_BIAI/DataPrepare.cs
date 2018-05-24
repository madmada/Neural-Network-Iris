using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


class DataPrepare
{
    public static double[] normalizeInput(double[] input, double[][] allData)
    {
        const int size = 4; //sepal length, sepal width, petal length, petal width
        double[][] normalizeInput = new double[allData.Length + 1][];
        for (int a = 0; a < allData.Length; ++a)
        {
            normalizeInput[a] = new double[size];
            for (int b = 0; b < size; ++b)
                normalizeInput[a][b] = allData[a][b];
        }
        normalizeInput[allData.Length] = new double[size];
        normalizeInput[allData.Length] = input;
        normalize(normalizeInput, new int[] { 0, 1, 2, 3 }); // first 4 columns only ! 
        return normalizeInput[allData.Length];
    }
    public static void makeTrainTest(double[][] allData, out double[][] trainData, out double[][] testData)
    {
        int rows = allData.Length;
        int columns = allData[0].Length; //7 columns
        int trainingRows = (int)(rows * 0.80); //80% trainData
        int testRows = rows - trainingRows; //20% testData

        //new arrays
        trainData = new double[trainingRows][];
        testData = new double[testRows][];

        int[] seq = new int[rows]; // create a random sequence of indexes
        for (int i = 0; i < seq.Length; ++i)
            seq[i] = i; // in cell is number of index for example tab[1] = 1 

        Random rnd = new Random(0);
        for (int i = 0; i < seq.Length; ++i)
        {
            int r = rnd.Next(i, seq.Length); //into r random number from ( 0 - how much data)
            int tmp = seq[r]; //saves values from random cell 

            seq[r] = seq[i];
            seq[i] = tmp;
        }

        int helper = 0; // index into sequence[]
        int j = 0; // index into trainData or testData

        //kopiuje 80% danych
        for (; helper < trainingRows; ++helper) // first rows to train data 
        {
            trainData[j] = new double[columns];
            int idx = seq[helper];
            Array.Copy(allData[idx], trainData[j], columns); //Array.Copy(old, copy, copy.Length);
            ++j;
        }

        j = 0; // reset to start of test data

        //copy 20% of data 
        for (; helper < rows; ++helper)
        {
            testData[j] = new double[columns];
            int idx = seq[helper];
            Array.Copy(allData[idx], testData[j], columns); //Array.Copy(old, copy, copy.Length);
            ++j;
        }
    }

    public static void normalize(double[][] dataMatrix, int[] cols)
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

