namespace IrisNeuralNetwork.NeuralNetwork;

/// <summary>One row of the training log (printed and exported to CSV).</summary>
public record TrainingLogEntry(int Epoch, double TrainLoss, double ValidationLoss, double ValidationAccuracy);

/// <summary>
/// 3-layer Multi-Layer Perceptron: Input -> Hidden -> Output,
/// trained with stochastic gradient descent + backpropagation.
/// </summary>
public class MultiLayerPerceptron
{
    private readonly InputLayer _input;
    private readonly HiddenLayer _hidden;
    private readonly OutputLayer _output;
    private readonly Random _random;

    public int InputSize => _input.Size;
    public int HiddenSize => _hidden.NeuronCount;
    public int OutputSize => _output.NeuronCount;

    public MultiLayerPerceptron(int inputSize, int hiddenSize, int outputSize, int seed = 42)
    {
        _random = new Random(seed);
        _input = new InputLayer(inputSize);
        _hidden = new HiddenLayer(inputSize, hiddenSize, _random);
        _output = new OutputLayer(hiddenSize, outputSize, _random);
    }

    /// <summary>Forward pass through all three layers.</summary>
    public double[] Predict(double[] features)
    {
        var x = _input.Forward(features);
        var h = _hidden.Forward(x);
        return _output.Forward(h);
    }

    /// <summary>Predicted class = index of the largest output neuron.</summary>
    public int PredictClass(double[] features)
    {
        var o = Predict(features);
        int best = 0;
        for (int k = 1; k < o.Length; k++)
            if (o[k] > o[best]) best = k;
        return best;
    }

    /// <summary>One SGD step on one sample. Returns that sample's loss.</summary>
    private double TrainSample(double[] features, double[] target, double learningRate)
    {
        Predict(features);                              // 1. forward pass
        double loss = _output.ComputeDelta(target);     // 2. output error
        _hidden.ComputeDelta(_output);                  // 3. hidden error (uses OLD output weights)
        _output.UpdateWeights(learningRate);            // 4. update weights
        _hidden.UpdateWeights(learningRate);
        return loss;
    }

    /// <summary>Mean loss over a dataset (no weight updates).</summary>
    public double ComputeLoss(double[][] features, double[][] targets)
    {
        double total = 0;
        for (int n = 0; n < features.Length; n++)
        {
            var o = Predict(features[n]);
            double sample = 0;
            for (int k = 0; k < o.Length; k++)
            {
                double e = o[k] - targets[n][k];
                sample += e * e;
            }
            total += sample / o.Length;
        }
        return total / features.Length;
    }

    public double ComputeAccuracy(double[][] features, int[] labels)
    {
        int correct = 0;
        for (int n = 0; n < features.Length; n++)
            if (PredictClass(features[n]) == labels[n]) correct++;
        return (double)correct / features.Length;
    }

    /// <summary>
    /// Trains the network. Samples are shuffled every epoch. Every
    /// <paramref name="logEvery"/> epochs (and at epoch 1) a log entry is recorded.
    /// </summary>
    public List<TrainingLogEntry> Train(
        double[][] trainX, double[][] trainY, int[] trainLabels,
        double[][] valX, double[][] valY, int[] valLabels,
        int epochs, double learningRate, int logEvery = 100, bool printLog = false)
    {
        var log = new List<TrainingLogEntry>();
        var order = Enumerable.Range(0, trainX.Length).ToArray();

        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            Shuffle(order);
            double epochLoss = 0;
            foreach (int i in order)
                epochLoss += TrainSample(trainX[i], trainY[i], learningRate);
            epochLoss /= trainX.Length;

            if (epoch == 1 || epoch % logEvery == 0 || epoch == epochs)
            {
                double valLoss = ComputeLoss(valX, valY);
                double valAcc = ComputeAccuracy(valX, valLabels);
                log.Add(new TrainingLogEntry(epoch, epochLoss, valLoss, valAcc));
                if (printLog)
                    Console.WriteLine($"Epoch {epoch,5} | Train Loss: {epochLoss:F6} | Val Loss: {valLoss:F6} | Val Acc: {valAcc * 100,6:F2}%");
            }
        }
        return log;
    }

    /// <summary>Fisher-Yates shuffle.</summary>
    private void Shuffle(int[] array)
    {
        for (int i = array.Length - 1; i > 0; i--)
        {
            int j = _random.Next(i + 1);
            (array[i], array[j]) = (array[j], array[i]);
        }
    }
}
