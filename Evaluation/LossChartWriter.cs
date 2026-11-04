using System.Globalization;
using System.Text;
using IrisNeuralNetwork.NeuralNetwork;

namespace IrisNeuralNetwork.Evaluation;

/// <summary>Visualisation helpers: an SVG loss curve file and a quick console bar chart.</summary>
public static class LossChartWriter
{
    /// <summary>Writes a standalone SVG line chart (open it in any browser).</summary>
    public static void WriteSvg(string path, List<TrainingLogEntry> log)
    {
        const int width = 800, height = 450, left = 70, right = 30, top = 50, bottom = 60;
        double maxLoss = Math.Max(log.Max(e => e.TrainLoss), log.Max(e => e.ValidationLoss)) * 1.05;
        int maxEpoch = log.Max(e => e.Epoch);
        double X(int epoch) => left + (double)epoch / maxEpoch * (width - left - right);
        double Y(double loss) => top + (1 - loss / maxLoss) * (height - top - bottom);
        string F(double v) => v.ToString("F1", CultureInfo.InvariantCulture);

        string Points(Func<TrainingLogEntry, double> sel) =>
            string.Join(" ", log.Select(e => $"{F(X(e.Epoch))},{F(Y(sel(e)))}"));

        var sb = new StringBuilder();
        sb.AppendLine($"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' font-family='Arial' font-size='12'>");
        sb.AppendLine($"<rect width='{width}' height='{height}' fill='white'/>");
        sb.AppendLine($"<text x='{width / 2}' y='28' text-anchor='middle' font-size='16' font-weight='bold'>Training Loss (MSE) vs Epoch</text>");

        for (int i = 0; i <= 5; i++)   // horizontal grid + y labels
        {
            double v = maxLoss * i / 5;
            sb.AppendLine($"<line x1='{left}' y1='{F(Y(v))}' x2='{width - right}' y2='{F(Y(v))}' stroke='#ddd'/>");
            sb.AppendLine($"<text x='{left - 8}' y='{F(Y(v) + 4)}' text-anchor='end'>{v.ToString("F3", CultureInfo.InvariantCulture)}</text>");
        }
        for (int i = 0; i <= 5; i++)   // x labels
        {
            int e = maxEpoch * i / 5;
            sb.AppendLine($"<text x='{F(X(e))}' y='{height - bottom + 20}' text-anchor='middle'>{e}</text>");
        }
        sb.AppendLine($"<text x='{width / 2}' y='{height - 15}' text-anchor='middle'>Epoch</text>");
        sb.AppendLine($"<line x1='{left}' y1='{top}' x2='{left}' y2='{height - bottom}' stroke='black'/>");
        sb.AppendLine($"<line x1='{left}' y1='{height - bottom}' x2='{width - right}' y2='{height - bottom}' stroke='black'/>");
        sb.AppendLine($"<polyline fill='none' stroke='#1f77b4' stroke-width='2' points='{Points(e => e.TrainLoss)}'/>");
        sb.AppendLine($"<polyline fill='none' stroke='#d62728' stroke-width='2' points='{Points(e => e.ValidationLoss)}'/>");
        sb.AppendLine($"<text x='{width - 190}' y='{top + 10}' fill='#1f77b4'>■ Train loss</text>");
        sb.AppendLine($"<text x='{width - 190}' y='{top + 28}' fill='#d62728'>■ Validation loss</text>");
        sb.AppendLine("</svg>");
        File.WriteAllText(path, sb.ToString());
    }

    /// <summary>Prints a simple horizontal bar chart of the training loss.</summary>
    public static void PrintConsoleChart(List<TrainingLogEntry> log)
    {
        double max = log.Max(e => e.TrainLoss);
        Console.WriteLine("Training loss curve (each # is a share of the first logged loss):");
        foreach (var e in log)
        {
            int bars = (int)Math.Round(e.TrainLoss / max * 50);
            Console.WriteLine($"{e.Epoch,5} | {new string('#', bars)} {e.TrainLoss:F5}");
        }
    }
}
