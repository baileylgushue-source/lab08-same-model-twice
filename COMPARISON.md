
# PyTorch vs. Keras Comparison

## Overview

In Lab 08, we built and trained the same neural network using PyTorch and Keras to predict bike rental demand. The goal was to compare their performance while keeping the model architecture, dataset, and training settings consistent.

## Model Architecture

Both models use:
- The same input features and preprocessing
- Two hidden layers with 16 neurons each
- ReLU activation functions
- One output neuron for predicting rentals
- Adam optimizer
- Mean Squared Error (MSE) loss
- Learning rate of 0.001
- Batch size of 64

Both models have 433 trainable parameters when the input size is 8 and the hidden size is 16.

Parameter counts were also verified for hidden sizes 8 and 32.

## Weight Equivalence

We copied the PyTorch model weights into the Keras model before training and compared their predictions.

The predictions matched within a tolerance of 1e-5, confirming that both networks represent the same function before training.

## Short Training Comparison

Both models were trained for 5 epochs using seed 42.

| Metric | PyTorch | Keras |
|---|---:|---:|
| Final training loss | 171.5349 | 172.1960 |
| Final validation loss | 194.4896 | 195.5255 |
| Best epoch | 5 | 5 |

Both models improved during training and produced similar results. PyTorch had a slightly lower validation loss in this short run.

## Three-Seed Comparison

Both frameworks were compared using seeds 42, 43, and 44.

| Seed | Framework | Best Epoch | Validation MAE | Test MAE | Training Time (s) |
|---|---|---:|---:|---:|---:|
| 42 | PyTorch | 100 | 6.6821 | 6.3971 | 4.1804 |
| 42 | Keras | 100 | 6.2426 | 6.0531 | 6.3385 |
| 43 | PyTorch | 100 | 6.4479 | 6.2413 | 2.3108 |
| 43 | Keras | 100 | 5.6469 | 5.5718 | 6.2090 |
| 44 | PyTorch | 100 | 5.9410 | 5.6197 | 2.4147 |
| 44 | Keras | 100 | 5.8167 | 5.2953 | 6.2885 |

## Results and Observations

Keras achieved a lower validation MAE and test MAE for all three seeds.

PyTorch trained faster in all three comparisons.

Both frameworks selected epoch 100 as their best epoch, meaning validation loss continued improving through the final epoch.

Although both models used the same architecture and training settings, their results were not identical. Differences in weight initialization and training implementation can affect the results.

## Learning Curves

The learning curves show that training and validation losses decreased over time for both frameworks.

The validation loss did not show a significant increase, so there was no clear evidence of overfitting in these runs.

The graphs are saved in:
- `reports/learning_curves.png`
- `reports/keras_learning_curves.png`

## Conclusion

Both PyTorch and Keras successfully trained the same neural network for predicting bike rental demand.

Keras produced slightly better prediction accuracy in these experiments, while PyTorch required less training time.

The weight-copy test confirmed that the two models could produce matching predictions before training. This demonstrates that differences in training results do not necessarily mean the model architectures are different.
