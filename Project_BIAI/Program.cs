using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

class Program
{
    private const int input = 4;
    private const int hidden = 7;
    private const int output = 3;

    static void Main(string[] args)
    {
        Console.WriteLine("****************** BIAI PROJECT - Neural Network ******************");
        Console.WriteLine("******************  Adam Pałka & Jakub Walczak  *******************\n");
        Console.WriteLine(" 0 0 1 = Iris Setosa");
        Console.WriteLine(" 0 1 0 = Iris Versicolor");
        Console.WriteLine(" 1 0 0 = Iris Virginica\n");
        Console.WriteLine("DATA: sepal length, sepal width, petal length, petal width");
        Console.WriteLine("INPUTA DATA example: 5.4, 3.4, 1.7, 0.2");
        Console.WriteLine("*******************************************************************\n\n");

        double[][] data = null;
        Tools.fileReader(ref data, @"..\..\DANE.txt"); //reading data from file
        double[][] dataForTest = null;
        double[][] dataForTraining = null;
        DataPrepare.makeTrainTest(data, out dataForTest, out dataForTraining);
        DataPrepare.normalize(dataForTest, new int[] { 0, 1, 2, 3 });
        DataPrepare.normalize(dataForTraining, new int[] { 0, 1, 2, 3 });

        Console.WriteLine("Creating neural network...");
        
        const int input = 4;
        const int hidden = 7;
        const int output = 3;
        NeuralNetwork nn = new NeuralNetwork(input, hidden, output);
        nn.InitializeWeights(); //Initializing weights and bias to small random values

        //TRAINING

        //Settings for training
        int epochs = 2000;
        double learnRate = 0.05;
        double momentum = 0.01;
        double weightDecay = 0.0001;
        Console.WriteLine("Start training");
        DateTime startTime = DateTime.Now;
        nn.Training(dataForTest, epochs, learnRate, momentum, weightDecay); 
        DateTime stopTime = DateTime.Now;
        TimeSpan roznica = stopTime - startTime;
        Console.WriteLine("Training time:" + roznica.TotalSeconds);
        Console.WriteLine("Training completed!");
        double[] weights = nn.GetWeights();

        //Console.WriteLine("nn weights and bias values:");
        //Tools.showVector(weights, 10, 3, true);

        //Accurancy
        double accurancyTrain = nn.accuracy(dataForTest);
        Console.WriteLine("--------------------Accurancy on Neural Network--------------------  ");
        Console.WriteLine("\nAccuracy on training data = " + accurancyTrain.ToString("F2"));
        double accurancyTest = nn.accuracy(dataForTraining);
        Console.WriteLine("Accuracy on test data = " + accurancyTest.ToString("F2"));

        //TESTING BY USER
        double[] numbers = new double[4];
        string userInput;
        Console.WriteLine("\n-------------------------------------------------------------------  ");
        Console.WriteLine("Would you like to test ? [Y/N]");
        userInput = Console.ReadLine();
        Tools.checkInputAnswer(userInput);       
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
                    Console.WriteLine(e.GetBaseException());
                }
                if (numbers[a] <= 0 )
                {
                    Console.WriteLine("Wrong Value!");
                    a--;
                }
            }
            Console.WriteLine(Tools.outputType(ref nn, DataPrepare.normalizeInput(numbers, data)));
            Console.WriteLine("Would you like to test again ? [Y/N]");
           
            userInput = Console.ReadLine();
            Tools.checkInputAnswer(userInput);
            
        }
    }
}

