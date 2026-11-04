using System.Globalization;
using System.Text;
using IrisNeuralNetwork.DataPipeline;
using IrisNeuralNetwork.Evaluation;
using IrisNeuralNetwork.Experiments;
using IrisNeuralNetwork.NeuralNetwork;

namespace IrisNeuralNetwork;

/// <summary>
/// Multi-class classification of Iris flowers using a custom 3-layer MLP
/// (Input -> Hidden -> Output) with backpropagation. No external ML libraries are used.
/// </summary>
public static class Program
{
    private const int Seed = 42;

    public static void Main()
    {
        CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
        CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
        string outputDir = Path.Combine(Directory.GetCurrentDirectory(), "Output");
        Directory.CreateDirectory(outputDir);

        // ---------- 1. LOAD DATA ----------
        PrintHeader("1. LOADING DATA");
        var dataset = CsvLoader.Load(FindDatasetPath());
        Console.WriteLine($"Samples: {dataset.Count} | Features: {string.Join(", ", dataset.FeatureNames)}");
        Console.WriteLine($"Classes: {string.Join(", ", dataset.ClassNames)}");

        // ---------- 2. SPLIT + SCALE ----------
        PrintHeader("2. SPLITTING (70/15/15 stratified) AND MIN-MAX SCALING");
        var split = DataSplitter.StratifiedSplit(dataset.Labels, 0.70, 0.15, Seed);
        Console.WriteLine($"Train: {split.Train.Length} | Validation: {split.Validation.Length} | Test: {split.Test.Length}");

        var trainRaw = DataSplitter.Select(dataset.Features, split.Train);
        var scaler = new MinMaxScaler();
        scaler.Fit(trainRaw);   // fit on training data only (prevents leakage)

        double[][] trainX = scaler.Transform(trainRaw);
        double[][] valX = scaler.Transform(DataSplitter.Select(dataset.Features, split.Validation));
        double[][] testX = scaler.Transform(DataSplitter.Select(dataset.Features, split.Test));

        int[] trainLabels = DataSplitter.Select(dataset.Labels, split.Train);
        int[] valLabels = DataSplitter.Select(dataset.Labels, split.Validation);
        int[] testLabels = DataSplitter.Select(dataset.Labels, split.Test);

        int classCount = dataset.ClassNames.Length;
        double[][] trainY = Dataset.OneHot(trainLabels, classCount);
        double[][] valY = Dataset.OneHot(valLabels, classCount);
        double[][] testY = Dataset.OneHot(testLabels, classCount);
        int inputSize = dataset.FeatureNames.Length;

        // ---------- 3. HYPERPARAMETER TUNING ----------
        PrintHeader("3. HYPERPARAMETER TUNING (grid search, scored on validation set)");
        double[] learningRates = { 0.01, 0.05, 0.1, 0.5 };
        int[] hiddenSizes = { 2, 4, 8, 16 };
        int[] epochOptions = { 100, 500, 1000 };

        var results = HyperparameterTuner.GridSearch(learningRates, hiddenSizes, epochOptions,
            trainX, trainY, trainLabels, valX, valY, valLabels, inputSize, classCount, Seed);
        HyperparameterTuner.SaveCsv(Path.Combine(outputDir, "tuning_results.csv"), results);
        var best = HyperparameterTuner.PickBest(results);

        Console.WriteLine();
        Console.WriteLine($"BEST CONFIG -> learning rate: {best.LearningRate}, hidden nodes: {best.HiddenNodes}, epochs: {best.Epochs} " +
                          $"(val acc {best.ValidationAccuracy * 100:F2}%, val loss {best.ValidationLoss:F5})");

        // ---------- 4. FINAL TRAINING WITH LOG ----------
        PrintHeader("4. FINAL TRAINING (loss printed every 100 epochs)");
        var model = new MultiLayerPerceptron(inputSize, best.HiddenNodes, classCount, Seed);
        var log = model.Train(trainX, trainY, trainLabels, valX, valY, valLabels,
            best.Epochs, best.LearningRate, logEvery: 100, printLog: true);
        SaveTrainingLog(Path.Combine(outputDir, "training_log.csv"), log);
        LossChartWriter.WriteSvg(Path.Combine(outputDir, "loss_curve.svg"), log);

        Console.WriteLine();
        LossChartWriter.PrintConsoleChart(log);

        // ---------- 5. EVALUATION ON UNSEEN TEST SET ----------
        PrintHeader("5. EVALUATION ON TEST SET (never used for training or tuning)");
        int[] predictions = testX.Select(model.PredictClass).ToArray();
        var metrics = new ClassificationMetrics(testLabels, predictions, dataset.ClassNames);
        Console.WriteLine($"Test MSE loss: {model.ComputeLoss(testX, testY):F6}");
        Console.WriteLine();
        string report = metrics.ToReport();
        Console.WriteLine(report);
        File.WriteAllText(Path.Combine(outputDir, "evaluation_report.txt"), report);
        SavePredictions(Path.Combine(outputDir, "test_predictions.csv"), model, testX, testLabels, dataset.ClassNames);

        Console.WriteLine($"Files saved to: {outputDir}");
        Console.WriteLine("  training_log.csv, tuning_results.csv, evaluation_report.txt, test_predictions.csv, loss_curve.svg");
    }

    private static string FindDatasetPath()
    {
        string[] candidates =
        {
            Path.Combine(AppContext.BaseDirectory, "Data", "iris.csv"),
            Path.Combine(Directory.GetCurrentDirectory(), "Data", "iris.csv")
        };
        return candidates.FirstOrDefault(File.Exists) ?? candidates[0];
    }

    private static void PrintHeader(string title)
    {
        Console.WriteLine();
        Console.WriteLine(new string('=', 70));
        Console.WriteLine(title);
        Console.WriteLine(new string('=', 70));
    }

    private static void SaveTrainingLog(string path, List<TrainingLogEntry> log)
    {
        var sb = new StringBuilder("epoch,train_loss,validation_loss,validation_accuracy\n");
        foreach (var e in log)
            sb.AppendLine($"{e.Epoch},{e.TrainLoss:F6},{e.ValidationLoss:F6},{e.ValidationAccuracy:F4}");
        File.WriteAllText(path, sb.ToString());
    }

    private static void SavePredictions(string path, MultiLayerPerceptron model, double[][] x, int[] actual, string[] classNames)
    {
        var sb = new StringBuilder("actual,predicted,correct,confidence\n");
        for (int i = 0; i < x.Length; i++)
        {
            var output = model.Predict(x[i]);
            int predicted = model.PredictClass(x[i]);
            sb.AppendLine($"{classNames[actual[i]]},{classNames[predicted]},{predicted == actual[i]},{output[predicted]:F4}");
        }
        File.WriteAllText(path, sb.ToString());
    }
}
