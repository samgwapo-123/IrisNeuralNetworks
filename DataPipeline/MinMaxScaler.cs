namespace IrisNeuralNetwork.DataPipeline;

/// <summary>
/// Min-Max scaling to [0,1]:  x' = (x - min) / (max - min).
/// Fit on the TRAINING set only, then applied to validation/test to avoid data leakage.
/// </summary>
public class MinMaxScaler
{
    private double[] _min = Array.Empty<double>();
    private double[] _max = Array.Empty<double>();

    public void Fit(double[][] data)
    {
        int cols = data[0].Length;
        _min = new double[cols];
        _max = new double[cols];
        for (int c = 0; c < cols; c++)
        {
            _min[c] = data.Min(r => r[c]);
            _max[c] = data.Max(r => r[c]);
        }
    }

    public double[][] Transform(double[][] data)
    {
        return data.Select(row =>
        {
            var scaled = new double[row.Length];
            for (int c = 0; c < row.Length; c++)
            {
                double range = _max[c] - _min[c];
                scaled[c] = range == 0 ? 0.0 : (row[c] - _min[c]) / range;
            }
            return scaled;
        }).ToArray();
    }
}
