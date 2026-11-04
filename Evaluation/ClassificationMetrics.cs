using System.Text;

namespace IrisNeuralNetwork.Evaluation;

/// <summary>Confusion matrix, accuracy, and per-class precision / recall / F1.</summary>
public class ClassificationMetrics
{
    public string[] ClassNames { get; }
    public int[,] Confusion { get; }   // [actual, predicted]
    public double Accuracy { get; }
    public double[] Precision { get; }
    public double[] Recall { get; }
    public double[] F1 { get; }

    public ClassificationMetrics(int[] actual, int[] predicted, string[] classNames)
    {
        ClassNames = classNames;
        int k = classNames.Length;
        Confusion = new int[k, k];
        for (int i = 0; i < actual.Length; i++)
            Confusion[actual[i], predicted[i]]++;

        Precision = new double[k];
        Recall = new double[k];
        F1 = new double[k];
        int correct = 0;

        for (int c = 0; c < k; c++)
        {
            int tp = Confusion[c, c];
            int fp = 0, fn = 0;
            for (int o = 0; o < k; o++)
            {
                if (o == c) continue;
                fp += Confusion[o, c];   // predicted c but was o
                fn += Confusion[c, o];   // was c but predicted o
            }
            correct += tp;
            Precision[c] = tp + fp == 0 ? 0 : (double)tp / (tp + fp);
            Recall[c] = tp + fn == 0 ? 0 : (double)tp / (tp + fn);
            F1[c] = Precision[c] + Recall[c] == 0 ? 0 : 2 * Precision[c] * Recall[c] / (Precision[c] + Recall[c]);
        }
        Accuracy = actual.Length == 0 ? 0 : (double)correct / actual.Length;
    }

    public double MacroPrecision => Precision.Average();
    public double MacroRecall => Recall.Average();
    public double MacroF1 => F1.Average();

    public string ToReport()
    {
        var sb = new StringBuilder();
        int k = ClassNames.Length;
        int w = Math.Max(12, ClassNames.Max(n => n.Length) + 2);

        sb.AppendLine("Confusion Matrix (rows = actual, columns = predicted)");
        sb.Append("".PadRight(w));
        foreach (var n in ClassNames) sb.Append(n.PadLeft(w));
        sb.AppendLine();
        for (int a = 0; a < k; a++)
        {
            sb.Append(ClassNames[a].PadRight(w));
            for (int p = 0; p < k; p++) sb.Append(Confusion[a, p].ToString().PadLeft(w));
            sb.AppendLine();
        }

        sb.AppendLine();
        sb.AppendLine($"{"Class".PadRight(w)}{"Precision",12}{"Recall",12}{"F1-score",12}");
        for (int c = 0; c < k; c++)
            sb.AppendLine($"{ClassNames[c].PadRight(w)}{Precision[c],12:F4}{Recall[c],12:F4}{F1[c],12:F4}");
        sb.AppendLine($"{"Macro avg".PadRight(w)}{MacroPrecision,12:F4}{MacroRecall,12:F4}{MacroF1,12:F4}");
        sb.AppendLine();
        sb.AppendLine($"Overall Accuracy: {Accuracy * 100:F2}%");
        return sb.ToString();
    }
}
