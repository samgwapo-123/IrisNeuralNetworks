namespace IrisNeuralNetwork.NeuralNetwork;

/// <summary>
/// Output layer (one neuron per class). Loss is Mean Squared Error:
/// L = 1/n * sum (target - output)^2   =>   dL/dout = 2/n * (output - target)
/// delta_k = dL/dout * sigmoid'(out_k)
/// </summary>
public class OutputLayer : DenseLayer
{
    public OutputLayer(int inputSize, int neuronCount, Random random)
        : base(inputSize, neuronCount, random) { }

    /// <summary>Computes the deltas and returns the MSE for this single sample.</summary>
    public double ComputeDelta(double[] target)
    {
        double loss = 0;
        for (int k = 0; k < NeuronCount; k++)
        {
            double error = Output[k] - target[k];
            loss += error * error;
            Delta[k] = (2.0 / NeuronCount) * error * SigmoidDerivative(Output[k]);
        }
        return loss / NeuronCount;
    }
}
