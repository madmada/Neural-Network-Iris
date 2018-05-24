using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


class Tools
{
    public static string outputType(ref NeuralNetwork nn, double[] numbers)
    {
        string ret = "";
        const int tot_len = 3;
        double[] results = nn.forward(numbers);
        for (int a = 0; a < tot_len; ++a)
        {
            if (results[a] + 0.15 > 1)
                ret += "1";
            else if (results[a] - 0.15 < 0)
                ret += "0";
            Console.WriteLine(results[a]); //printing real output of nn 
        }

        if (ret == "001")
            return "Iris Setosa";
        else if (ret == "010")
            return "Iris Versicolor";
        else if (ret == "100")
            return "Iris Virginica";
        else
        {
            ret = "";
            for (int a = 0; a < tot_len; ++a)
            {
                if (results[a] + 0.35 > 1)
                    ret += "1";
                else if (results[a] - 0.35 < 0)
                    ret += "0";
            }

            if (ret == "001")
                return "(PROBABLY) : Iris Setosa";
            else if (ret == "010")
                return "(PROBABLY) : Iris Versicolor";
            else if (ret == "100")
                return "(PROBABLY) : Iris Virginica";
            else
            {
                return "Iris type unknown - further training needed";
            }
        }
    }
    public static void fileReader(ref double[][] allData, string path)
    {

        const int size_2d = 7;
        string[] splitchar = new string[] { ", " };
        string[] lines = System.IO.File.ReadAllLines(path);
        string[][] splitlines = new string[lines.Length][];
        allData = new double[lines.Length][];

        for (int a = 0; a < lines.Length; ++a)
        {
            splitlines[a] = new string[size_2d];
            allData[a] = new double[size_2d];
        }
        for (int a = 0; a < lines.Length; ++a)
            splitlines[a] = lines[a].Split(splitchar, StringSplitOptions.None);

        for (int a = 0; a < lines.Length; ++a)
            for (int b = 0; b < size_2d; ++b)
                allData[a][b] = Convert.ToDouble(splitlines[a][b], System.Globalization.CultureInfo.InvariantCulture);

    }

    public static void showMatrix(double[][] matrix, int numRows, int decimals, bool newLine)
    {
        for (int i = 0; i < numRows; ++i)
        {
            Console.Write(i.ToString().PadLeft(3) + ": ");
            for (int j = 0; j < matrix[i].Length; ++j)
            {
                if (matrix[i][j] >= 0.0) Console.Write(" "); else Console.Write("-");
                Console.Write(Math.Abs(matrix[i][j]).ToString("F" + decimals) + " ");
            }
            Console.WriteLine("");
        }
        if (newLine == true) Console.WriteLine("");
    }

    public static void showVector(double[] vector, int valsPerRow, int decimals, bool newLine)
    {
        for (int i = 0; i < vector.Length; ++i)
        {
            if (i % valsPerRow == 0) Console.WriteLine("");
            Console.Write(vector[i].ToString("F" + decimals).PadLeft(decimals + 4) + " ");
        }
        if (newLine == true) Console.WriteLine("");
    }

    // Check if user entered correct answer (Y - yes or N - no)
    public static void checkInputAnswer(string inputText)
    {
        while (inputText.ToUpper() != "Y" && inputText.ToUpper() != "N")
        {
            Console.WriteLine("Please enter correct answer [Y/N]");
            inputText = Console.ReadLine();
        }
    }
}

