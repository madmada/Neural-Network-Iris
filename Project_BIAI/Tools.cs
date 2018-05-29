using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;

class Tools
{
    public static string OutputType(ref NeuralNetwork neuralNetwork, double[] numbers)
    {
        string result = "";
        const int length = 3;
        double hardPrecision = 0.15;
        double softPrecision = 0.35;
        double[] nnResults = neuralNetwork.Forward(numbers);
        for (int i = 0; i < length; ++i)
        {
            if (nnResults[i] + hardPrecision > 1)
                result += "1";
            else if (nnResults[i] - hardPrecision < 0)
                result += "0";
            Console.WriteLine(nnResults[i]); //printing real output of neural network 
        }

        if (result == "001")
            return "Iris Setosa";
        else if (result == "010")
            return "Iris Versicolor";
        else if (result == "100")
            return "Iris Virginica";
        else
        {
            result = "";
            for (int i = 0; i < length; ++i)
            {
                if (nnResults[i] + softPrecision > 1)
                    result += "1";
                else if (nnResults[i] - softPrecision < 0)
                    result += "0";
            }

            if (result == "001")
                return "(PROBABLY) : Iris Setosa";
            else if (result == "010")
                return "(PROBABLY) : Iris Versicolor";
            else if (result == "100")
                return "(PROBABLY) : Iris Virginica";
            else
                return "Iris type unknown - further training needed";
        }
    }
    public static void FileReader(ref double[][] allData, string path)
    {
        const int size_2d = 7;
        string[] splitchar = new string[] { ", " };
        string[] lines = System.IO.File.ReadAllLines(path);
        string[][] splitlines = new string[lines.Length][];
        allData = new double[lines.Length][];

        for (int i = 0; i < lines.Length; ++i)
        {
            splitlines[i] = new string[size_2d];
            allData[i] = new double[size_2d];
        }
        for (int i = 0; i < lines.Length; ++i)
            splitlines[i] = lines[i].Split(splitchar, StringSplitOptions.None);

        try
        {
            for (int i = 0; i < lines.Length; ++i)
                for (int j = 0; j < size_2d; ++j)
                    allData[i][j] = Convert.ToDouble(splitlines[i][j], System.Globalization.CultureInfo.InvariantCulture);
        }
        catch (Exception e)
        {
            Console.WriteLine("Error occurred during loading data from file.");
            Thread.Sleep(10000);
            Environment.Exit(1);
        }

    }
    // Check if user entered correct answer (Y - yes or N - no)
    public static void CheckInputAnswer(string inputText)
    {
        while (inputText.ToUpper() != "Y" && inputText.ToUpper() != "N")
        {
            Console.WriteLine("Please enter correct answer [Y/N]");
            inputText = Console.ReadLine();
        }
    }
    //public static void ShowMatrix(double[][] matrix, int numRows, int decimals, bool newLine)
    //{
    //    for (int i = 0; i < numRows; ++i)
    //    {
    //        Console.Write(i.ToString().PadLeft(3) + ": ");
    //        for (int j = 0; j < matrix[i].Length; ++j)
    //        {
    //            if (matrix[i][j] >= 0.0) Console.Write(" "); else Console.Write("-");
    //            Console.Write(Math.Abs(matrix[i][j]).ToString("F" + decimals) + " ");
    //        }
    //        Console.WriteLine("");
    //    }
    //    if (newLine == true) Console.WriteLine("");
    //}
    //public static void ShowVector(double[] vector, int valsPerRow, int decimals, bool newLine)
    //{
    //    for (int i = 0; i < vector.Length; ++i)
    //    {
    //        if (i % valsPerRow == 0) Console.WriteLine("");
    //        Console.Write(vector[i].ToString("F" + decimals).PadLeft(decimals + 4) + " ");
    //    }
    //    if (newLine == true) Console.WriteLine("");
    //}
}

