namespace IrisNeuralNetwork.DataPipeline;

/// <summary>Result of splitting: index lists into the original dataset.</summary>
public record SplitIndices(int[] Train, int[] Validation, int[] Test);

/// <summary>
/// Stratified shuffle split: every class keeps (roughly) the same proportion in
/// the train, validation and test sets.
/// </summary>
public static class DataSplitter
{
    public static SplitIndices StratifiedSplit(int[] labels, double trainRatio, double validationRatio, int seed)
    {
        var random = new Random(seed);
        var train = new List<int>();
        var validation = new List<int>();
        var test = new List<int>();

        foreach (var group in labels.Select((label, index) => (label, index)).GroupBy(x => x.label))
        {
            var indices = group.Select(x => x.index).OrderBy(_ => random.Next()).ToList();
            int trainCount = (int)Math.Round(indices.Count * trainRatio);
            int valCount = (int)Math.Round(indices.Count * validationRatio);

            train.AddRange(indices.Take(trainCount));
            validation.AddRange(indices.Skip(trainCount).Take(valCount));
            test.AddRange(indices.Skip(trainCount + valCount));
        }

        return new SplitIndices(train.ToArray(), validation.ToArray(), test.ToArray());
    }

    public static T[] Select<T>(T[] source, int[] indices) => indices.Select(i => source[i]).ToArray();
}
