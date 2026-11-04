using System.Globalization;
using System.Text;
using IrisNeuralNetwork.NeuralNetwork;

namespace IrisNeuralNetwork.Experiments;

public record TuningResult(double LearningRate, int HiddenNodes, int Epochs,
                           double TrainLoss, double ValidationLoss, double ValidationAccuracy);

/// <summary>
/// Grid search over learning rate, hidden-layer size and epoch count.
/// Models are compared on the VALIDATION set only, so the test set stays untouched
/// until the very end.
/// </summary>
public static class HyperparameterTuner
{
    public static List<TuningResult> GridSearch(
        double[] learningRates, int[] hiddenSizes, int[] epochOptions,
        double[][] trainX, double[][] trainY, int[] trainLabels,
        double[][] valX, double[][] valY, int[] valLabels,
        int inputSize, int outputSize, int seed)
    {
        var results = new List<TuningResult>();
        int total = learningRates.Length * hiddenSizes.Length * epochOptions.Length;
        int done = 0;

        Console.WriteLine($"{"LR",8}{"Hidden",8}{"Epochs",8}{"TrainLoss",12}{"ValLoss",12}{"ValAcc",10}");
        foreach (var lr in learningRates)
        foreach (var hidden in hiddenSizes)
        foreach (var epochs in epochOptions)
        {
            var model = new MultiLayerPerceptron(inputSize, hidden, outputSize, seed);
            var log = model.Train(trainX, trainY, trainLabels, valX, valY, valLabels, epochs, lr, logEvery: epochs);
            var last = log[^1];
            var r = new TuningResult(lr, hidden, epochs, last.TrainLoss, last.ValidationLoss, last.ValidationAccuracy);
            results.Add(r);
            done++;
            Console.WriteLine($"{lr,8}{hidden,8}{epochs,8}{r.TrainLoss,12:F5}{r.ValidationLoss,12:F5}{r.ValidationAccuracy * 100,9:F2}%   ({done}/{total})");
        }
        return results;
    }

    /// <summary>Best = highest validation accuracy; ties broken by lowest validation loss.</summary>
    public static TuningResult PickBest(List<TuningResult> results) =>
        results.OrderByDescending(r => r.ValidationAccuracy).ThenBy(r => r.ValidationLoss).First();

    public static void SaveCsv(string path, List<TuningResult> results)
    {
        var sb = new StringBuilder("learning_rate,hidden_nodes,epochs,train_loss,validation_loss,validation_accuracy\n");
        foreach (var r in results)
            sb.AppendLine(string.Join(",",
                r.LearningRate.ToString(CultureInfo.InvariantCulture), r.HiddenNodes, r.Epochs,
                r.TrainLoss.ToString("F6", CultureInfo.InvariantCulture),
                r.ValidationLoss.ToString("F6", CultureInfo.InvariantCulture),
                r.ValidationAccuracy.ToString("F4", CultureInfo.InvariantCulture)));
        File.WriteAllText(path, sb.ToString());
    }
}
