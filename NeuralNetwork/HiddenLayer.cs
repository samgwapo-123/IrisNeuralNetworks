namespace IrisNeuralNetwork.NeuralNetwork;

/// <summary>
/// Hidden layer. Its error term comes from the layer AFTER it:
/// delta_j = (sum_k delta_k * w_jk) * sigmoid'(out_j)
/// </summary>
public class HiddenLayer : DenseLayer
{
    public HiddenLayer(int inputSize, int neuronCount, Random random)
        : base(inputSize, neuronCount, random) { }

    public void ComputeDelta(DenseLayer nextLayer)
    {
        for (int j = 0; j < NeuronCount; j++)
            Delta[j] = nextLayer.BackpropagatedError(j) * SigmoidDerivative(Output[j]);
    }
}
