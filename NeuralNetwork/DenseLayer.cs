namespace IrisNeuralNetwork.NeuralNetwork;

/// <summary>
/// Shared logic for a fully-connected layer with sigmoid activation.
/// HiddenLayer and OutputLayer inherit from this and only differ in how
/// their error term (delta) is computed during backpropagation.
/// </summary>
public abstract class DenseLayer
{
    public int InputSize { get; }
    public int NeuronCount { get; }

    protected readonly double[,] Weights;   // [inputIndex, neuronIndex]
    protected readonly double[] Biases;     // [neuronIndex]

    protected double[] LastInput = Array.Empty<double>();
    public double[] Output { get; protected set; }

    /// <summary>Error term (dLoss/dNet) for each neuron, filled during backpropagation.</summary>
    public double[] Delta { get; protected set; }

    protected DenseLayer(int inputSize, int neuronCount, Random random)
    {
        InputSize = inputSize;
        NeuronCount = neuronCount;
        Weights = new double[inputSize, neuronCount];
        Biases = new double[neuronCount];
        Output = new double[neuronCount];
        Delta = new double[neuronCount];

        // Xavier/Glorot uniform initialisation keeps early activations in a healthy range.
        double limit = Math.Sqrt(6.0 / (inputSize + neuronCount));
        for (int i = 0; i < inputSize; i++)
            for (int j = 0; j < neuronCount; j++)
                Weights[i, j] = (random.NextDouble() * 2 - 1) * limit;
    }

    protected static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

    /// <summary>Derivative of sigmoid expressed in terms of the sigmoid OUTPUT.</summary>
    protected static double SigmoidDerivative(double output) => output * (1.0 - output);

    /// <summary>Forward pass: output = sigmoid(input · weights + bias).</summary>
    public double[] Forward(double[] input)
    {
        LastInput = input;
        for (int j = 0; j < NeuronCount; j++)
        {
            double sum = Biases[j];
            for (int i = 0; i < InputSize; i++)
                sum += input[i] * Weights[i, j];
            Output[j] = Sigmoid(sum);
        }
        return Output;
    }

    /// <summary>
    /// Used by the previous layer: sum over this layer's neurons of (delta * weight)
    /// for one input index. This is how the error flows backwards.
    /// </summary>
    public double BackpropagatedError(int inputIndex)
    {
        double sum = 0;
        for (int j = 0; j < NeuronCount; j++)
            sum += Delta[j] * Weights[inputIndex, j];
        return sum;
    }

    /// <summary>Gradient-descent step using the stored Delta and the last input.</summary>
    public void UpdateWeights(double learningRate)
    {
        for (int j = 0; j < NeuronCount; j++)
        {
            for (int i = 0; i < InputSize; i++)
                Weights[i, j] -= learningRate * Delta[j] * LastInput[i];
            Biases[j] -= learningRate * Delta[j];
        }
    }
}
