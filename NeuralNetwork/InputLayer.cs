namespace IrisNeuralNetwork.NeuralNetwork;

/// <summary>
/// Input layer: holds the feature vector and passes it on unchanged.
/// It has no weights, it simply represents the raw (scaled) features.
/// </summary>
public class InputLayer
{
    public int Size { get; }
    public double[] Values { get; private set; }

    public InputLayer(int size)
    {
        Size = size;
        Values = new double[size];
    }

    public double[] Forward(double[] features)
    {
        if (features.Length != Size)
            throw new ArgumentException($"Expected {Size} features but received {features.Length}.");
        Values = features;
        return Values;
    }
}
