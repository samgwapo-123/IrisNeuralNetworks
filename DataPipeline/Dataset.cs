namespace IrisNeuralNetwork.DataPipeline;

/// <summary>Holds parsed features, integer class labels and their names.</summary>
public class Dataset
{
    public string[] FeatureNames { get; }
    public string[] ClassNames { get; }
    public double[][] Features { get; }
    public int[] Labels { get; }

    public int Count => Features.Length;

    public Dataset(string[] featureNames, string[] classNames, double[][] features, int[] labels)
    {
        FeatureNames = featureNames;
        ClassNames = classNames;
        Features = features;
        Labels = labels;
    }

    /// <summary>Converts integer labels to one-hot vectors, e.g. 1 of 3 classes -> [0,1,0].</summary>
    public static double[][] OneHot(int[] labels, int classCount)
    {
        var result = new double[labels.Length][];
        for (int i = 0; i < labels.Length; i++)
        {
            result[i] = new double[classCount];
            result[i][labels[i]] = 1.0;
        }
        return result;
    }
}
