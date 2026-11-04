# Iris Flower Classification — Custom C# Multi-Layer Perceptron

**Track:** Multi-Class Classification
**Language / Runtime:** C# on .NET 8 (no external ML or NuGet packages)

A 3-layer neural network (Input -> Hidden -> Output) with backpropagation, written from scratch,
that classifies iris flowers into 3 species from 4 measurements.

## How to run

1. Install the [.NET 8 SDK](https://dotnet.microsoft.com/download) (or newer).
2. Open a terminal in this folder (the one containing `IrisNeuralNetwork.csproj`).
3. Run:
   ```
   dotnet run
   ```
   Or open the `.csproj` in Visual Studio / Rider and press Run.

Results are written to an `Output/` folder: `training_log.csv`, `tuning_results.csv`,
`evaluation_report.txt`, `test_predictions.csv`, `loss_curve.svg` (open in a browser).
A copy of a finished run is in `Sample_Output/`.

## Project structure

| Folder / File | Purpose |
|---|---|
| `NeuralNetwork/InputLayer.cs` | Input layer (feature vector holder) |
| `NeuralNetwork/HiddenLayer.cs` | Hidden layer, error term from the next layer |
| `NeuralNetwork/OutputLayer.cs` | Output layer, MSE loss + output error term |
| `NeuralNetwork/DenseLayer.cs` | Shared forward pass / weight update / sigmoid logic |
| `NeuralNetwork/MultiLayerPerceptron.cs` | Wires the 3 layers; training loop (SGD + backprop) |
| `DataPipeline/` | CSV parsing, missing-value handling, Min-Max scaler, stratified split |
| `Experiments/HyperparameterTuner.cs` | Grid search over learning rate, hidden nodes, epochs |
| `Evaluation/` | Confusion matrix, precision/recall/F1, SVG loss chart |
| `Data/iris.csv` | Dataset |
| `Program.cs` | Runs the whole pipeline in order |

## Math summary

- Activation: sigmoid `σ(x) = 1 / (1 + e^-x)`, derivative `σ'(out) = out · (1 - out)`
- Loss (per sample): `MSE = (1/n) Σ (output_k - target_k)²`
- Output delta: `δ_k = (2/n)(out_k - t_k) · σ'(out_k)`
- Hidden delta: `δ_j = (Σ_k δ_k · w_jk) · σ'(out_j)`
- Update: `w -= learningRate · δ · input`, `b -= learningRate · δ`
- Targets are one-hot encoded; predicted class = output neuron with the highest value.
- Weights use Xavier uniform initialisation; samples are shuffled every epoch (SGD).

## Pipeline

1. Parse CSV (missing values -> column mean; malformed rows skipped).
2. Stratified split: 70% train / 15% validation / 15% test (fixed seed 42).
3. Min-Max scale features to [0,1]. The scaler is fitted on the **training set only** (no leakage).
4. Grid search: learning rate {0.01, 0.05, 0.1, 0.5} x hidden nodes {2, 4, 8, 16} x epochs {100, 500, 1000} = 48 runs,
   ranked on the **validation** set (accuracy, then validation loss).
5. Retrain the best configuration, logging loss every 100 epochs.
6. Evaluate once on the untouched **test** set.

## Hyperparameter findings

(Full table: `Output/tuning_results.csv`.)

- **Learning rate:** 0.01 is too slow; after 1000 epochs the loss was still ~0.08. 0.05–0.1 converge well.
  0.5 converged fastest (good by epoch 100) and stayed stable with sigmoid units.
  (An earlier run with 1.0 worked but gave noisier training loss, so it was dropped from the grid.)
- **Hidden nodes:** 2 nodes already work but learn slower/less smoothly; 4–16 are all similar.
  8 nodes gave the lowest validation loss at the best learning rate, with no benefit from 16 (no need for more capacity).
- **Epochs:** 100 is not enough at small learning rates; 500–1000 reach near-zero validation loss at higher rates.
- **Chosen configuration:** learning rate 0.5, 8 hidden nodes, 1000 epochs.

## Results (test set)

Accuracy 100% with perfect precision, recall and F1 for all three classes (see `Sample_Output/evaluation_report.txt`).

Notes for the analysis / report:
- The test set is only 21 flowers (7 per class), so 100% should not be read as "perfect in general";
  with so little data one different flower would change the number.
- The training loss levels off around 0.015 instead of 0 because a couple of *versicolor* and *virginica*
  training samples genuinely overlap in feature space (a known property of Iris), so no smooth model fits them exactly.
  Setosa is linearly separable from the other two.
- Validation/test loss is lower than training loss because training loss is averaged *while* weights are changing
  during the epoch, and the small validation/test sets happen to contain mostly clear-cut flowers.

## Using your own layer classes from the last activity

The `NeuralNetwork/` folder contains this project's own Input/Hidden/Output classes so it runs out of the box.
If your instructor requires your exact classes from the previous activity, replace the files in `NeuralNetwork/`
and keep the `MultiLayerPerceptron` public methods (`Train`, `Predict`, `PredictClass`, `ComputeLoss`, `ComputeAccuracy`).
