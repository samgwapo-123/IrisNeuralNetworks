using System.Globalization;

namespace IrisNeuralNetwork.DataPipeline;

/// <summary>
/// Parses a CSV whose last column is the class label and every other column is a numeric feature.
/// Missing values (empty, "NA", "?", "NaN") are replaced by the column mean.
/// Rows with the wrong number of columns or non-numeric junk are skipped with a warning.
/// </summary>
public static class CsvLoader
{
    private static readonly HashSet<string> MissingTokens =
        new(StringComparer.OrdinalIgnoreCase) { "", "NA", "N/A", "?", "NaN", "null" };

    public static Dataset Load(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"Dataset not found: {path}");

        var lines = File.ReadAllLines(path).Where(l => !string.IsNullOrWhiteSpace(l)).ToList();
        if (lines.Count < 2) throw new InvalidDataException("CSV needs a header and at least one row.");

        var header = lines[0].Split(',').Select(h => h.Trim()).ToArray();
        int featureCount = header.Length - 1;
        var featureNames = header.Take(featureCount).ToArray();

        var rawFeatures = new List<double[]>();
        var rawLabels = new List<string>();
        int skipped = 0;

        for (int line = 1; line < lines.Count; line++)
        {
            var parts = lines[line].Split(',').Select(p => p.Trim()).ToArray();
            if (parts.Length != header.Length || string.IsNullOrEmpty(parts[^1]))
            {
                skipped++;
                continue;
            }

            var row = new double[featureCount];
            bool valid = true;
            for (int c = 0; c < featureCount; c++)
            {
                if (MissingTokens.Contains(parts[c])) row[c] = double.NaN;
                else if (double.TryParse(parts[c], NumberStyles.Float, CultureInfo.InvariantCulture, out double v)) row[c] = v;
                else { valid = false; break; }
            }
            if (!valid) { skipped++; continue; }

            rawFeatures.Add(row);
            rawLabels.Add(parts[^1]);
        }

        if (skipped > 0) Console.WriteLine($"[CsvLoader] Skipped {skipped} malformed row(s).");

        ImputeMissingWithColumnMean(rawFeatures, featureCount);

        var classNames = rawLabels.Distinct().OrderBy(x => x, StringComparer.Ordinal).ToArray();
        var labels = rawLabels.Select(l => Array.IndexOf(classNames, l)).ToArray();

        return new Dataset(featureNames, classNames, rawFeatures.ToArray(), labels);
    }

    private static void ImputeMissingWithColumnMean(List<double[]> rows, int featureCount)
    {
        for (int c = 0; c < featureCount; c++)
        {
            var known = rows.Where(r => !double.IsNaN(r[c])).Select(r => r[c]).ToList();
            double mean = known.Count > 0 ? known.Average() : 0.0;
            int filled = 0;
            foreach (var r in rows)
                if (double.IsNaN(r[c])) { r[c] = mean; filled++; }
            if (filled > 0) Console.WriteLine($"[CsvLoader] Filled {filled} missing value(s) in column {c} with mean {mean:F3}.");
        }
    }
}
