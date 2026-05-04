```markdown
# KNN Classification on CIFAR-10

## File Structure

| File | Description |
|------|-------------|
| `knn_cifar10.py` | sklearn-based KNN — Train/Test split, Train/Val/Test split, 5-Fold CV |
| `knn_scratch_cifar10.py` | KNN from scratch — L1/L2 distance manually implemented, 5-Fold CV |
| `knn_result.png` | Result graph from knn_cifar10.py |
| `knn_scratch_results.png` | Result graph from knn_scratch_cifar10.py |
| `data/` | CIFAR-10 dataset directory |

## How to Run

```bash
pip install numpy matplotlib scikit-learn torchvision torch

python knn_cifar10.py
python knn_scratch_cifar10.py
```

## Dataset Note

At the time of execution (2026-05-05), the University of Toronto server
(`https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz`) was
unavailable due to a scheduled power outage, making CIFAR-10 download impossible.

- `knn_cifar10.py`: Uses synthetic dataset generated via `sklearn.make_classification`
  (10 classes, 2000 samples, 50 features, class_sep=0.5)
- `knn_scratch_cifar10.py`: Uses Gaussian-based surrogate dataset
  (10 classes, train 5000 / test 1000, 3072 features mimicking 32×32×3)

Once torchvision is properly installed, both scripts automatically
switch to real CIFAR-10 data. Expected KNN accuracy on real CIFAR-10
using raw pixels is approximately **25–35%**.

## Results Summary

### knn_cifar10.py (sklearn, synthetic data)
- Best k selected by 5-Fold CV: k ≈ 13–15
- CV Accuracy: ~0.42
- Accuracy increases with k up to ~k=15, then plateaus

### knn_scratch_cifar10.py (from scratch, surrogate data)
- Best config: L2 distance, k=1, Acc=0.216
- L2 consistently outperforms L1 across all k values
- Performance slightly decreases as k increases

## Limitations of KNN for Image Classification

1. **Computational Cost** — O(N·D) per query. No training phase;
   all 5,000 training samples must be compared at inference time.

2. **Curse of Dimensionality** — In 3,072-dimensional pixel space,
   distances become nearly uniform, making nearest neighbors unreliable.

3. **Pixel Distance ≠ Semantic Similarity** — L1/L2 distances are
   sensitive to brightness, translation, and background variation.
   Two images of the same class can be farther apart than two
   different classes with similar color distributions.

4. **No Feature Learning** — KNN uses raw pixels with no learned
   representation. CNNs learn hierarchical features and achieve
   ~93% accuracy vs KNN's ~25–35% on CIFAR-10.

5. **Memory Inefficiency** — The entire training set must be stored
   and queried at inference time, which is impractical at scale.

## Environment

- Python 3.12
- scikit-learn 1.8.0
- matplotlib 3.10.8
- torch 2.11.0 / torchvision 0.26.0
```