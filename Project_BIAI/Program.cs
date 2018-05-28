using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("****************** BIAI PROJECT - Neural Network ******************");
        Console.WriteLine("******************  Adam Pałka & Jakub Walczak  *******************\n");
        //Console.WriteLine(" 0 0 1 = Iris Setosa");
        //Console.WriteLine(" 0 1 0 = Iris Versicolor");
        //Console.WriteLine(" 1 0 0 = Iris Virginica\n");
        Console.WriteLine("INPUT DATA: sepal length, sepal width, petal length, petal width");
        Console.WriteLine("INPUT DATA example");
        Console.WriteLine("5.1, 3.5, 1.4, 0.2 (Iris Setosa)");
        Console.WriteLine("7.0, 3.2, 4.7, 1.4 (Iris Versicolor)");
        Console.WriteLine("6.8, 3.0, 5.5, 2.1 (Iris Virginica)");
        Console.WriteLine("*******************************************************************\n\n");

        double[][] data = null;
        double[][] dataForTest = null;
        double[][] dataForTraining = null;

        Tools.FileReader(ref data, @"..\..\DATASET.txt"); //reading data from file
        
        DataPrepare.MakeTrainTest(data, ref dataForTest, ref dataForTraining);

        DataPrepare.Normalize(dataForTest, new int[] { 0, 1, 2, 3 });

        DataPrepare.Normalize(dataForTraining, new int[] { 0, 1, 2, 3 });

        Console.WriteLine("Creating neural network...");
        
        const int input = 4;
        const int hidden = 7;
        const int output = 3;
        NeuralNetwork neuralNetwork = new NeuralNetwork(input, hidden, output);
        neuralNetwork.InitializeWeights(); //Initializing weights and bias to small random values

        //TRAINING

        //Settings for training
        int epochos = 2000;
        double learnRate = 0.05;
        double momentum = 0.01;
        double weightDecay = 0.0001;
        Console.WriteLine("Start training");
        DateTime startTime = DateTime.Now;
        neuralNetwork.Training(dataForTraining, epochos, learnRate, momentum, weightDecay); 
        DateTime stopTime = DateTime.Now;
        TimeSpan timeDifference = stopTime - startTime;
        Console.WriteLine("Training time:" + timeDifference.TotalSeconds);
        Console.WriteLine("Training completed!");
        double[] weights = neuralNetwork.GetWeights();
        
        //Accuracy
        double accurancyTrain = neuralNetwork.Accuracy(dataForTraining);
        Console.WriteLine("--------------------Accurancy on Neural Network--------------------  ");
        Console.WriteLine("\nAccuracy on training data = " + accurancyTrain.ToString("F2"));
        double accurancyTest = neuralNetwork.Accuracy(dataForTest);
        Console.WriteLine("Accuracy on test data = " + accurancyTest.ToString("F2"));

        //TESTING BY USER
        double[] numbers = new double[4];
        string userInput;
        Console.WriteLine("\n-------------------------------------------------------------------  ");
        Console.WriteLine("Would you like to test ? [Y/N]");
        userInput = Console.ReadLine();
        Tools.CheckInputAnswer(userInput);       
        while (userInput.ToUpper() == "Y")
        {
            Console.WriteLine("Please enter 4 values [sepal length, sepal width, petal length, petal width]: ");
            for (int a = 0; a < 4; a++)
            {
                try
                {
                    numbers[a] = Convert.ToDouble(Console.ReadLine(), System.Globalization.CultureInfo.InvariantCulture);
                }
                catch(Exception e)
                {
                    Console.WriteLine("Wrong value, try again!");
                }
                if (numbers[a] <= 0 )
                {
                    Console.WriteLine("Wrong value, try again!");
                    a--;
                }
            }
            Console.WriteLine(Tools.OutputType(ref neuralNetwork, DataPrepare.NormalizeInput(numbers, data)));
            Console.WriteLine("Would you like to test again ? [Y/N]");
           
            userInput = Console.ReadLine();
            Tools.CheckInputAnswer(userInput);
            
        }
    }
}

