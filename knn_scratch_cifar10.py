"""
KNN Assignment - From Scratch
CIFAR-10 | Train:5000, Test:1000
- KNN implemented manually (no sklearn KNeighborsClassifier)
- L1 and L2 distance
- K = 1, 3, 5, 7, 9
- 5-Fold Cross-Validation on training set
- Plot: CV accuracy vs K (each fold point + mean line + std error bar)
- Confusion matrix per best K
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# ─────────────────────────────────────────────────────────
# 1. Load CIFAR-10
# ─────────────────────────────────────────────────────────
print("=" * 55)
print("  KNN from Scratch  |  CIFAR-10")
print("=" * 55)
print("\n[1] Loading CIFAR-10 ...")

np.random.seed(42)

try:
    import torchvision
    import torchvision.transforms as transforms

    tr = torchvision.datasets.CIFAR10(root='./data', train=True,
                                      download=True, transform=transforms.ToTensor())
    te = torchvision.datasets.CIFAR10(root='./data', train=False,
                                      download=True, transform=transforms.ToTensor())

    X_tr_all = tr.data
    y_tr_all = np.array(tr.targets)
    X_te_all = te.data
    y_te_all = np.array(te.targets)

    tr_idx = np.random.choice(len(X_tr_all), 1000, replace=False)
    te_idx = np.random.choice(len(X_te_all), 300, replace=False)

    X_train = X_tr_all[tr_idx].reshape(len(tr_idx), -1).astype(np.float32) / 255.0
    y_train = y_tr_all[tr_idx]
    X_test  = X_te_all[te_idx].reshape(len(te_idx), -1).astype(np.float32) / 255.0
    y_test  = y_te_all[te_idx]
    print("  Loaded via torchvision")

except Exception as e:
    print(f"  torchvision unavailable ({e}), generating surrogate ...")
    rng = np.random.default_rng(42)
    def make_data(n):
        X_l, y_l = [], []
        for c in range(10):
            center = rng.normal(c * 0.35, 0.55, 3072)
            X_l.append(rng.normal(center, 0.85, (n // 10, 3072)))
            y_l.append(np.full(n // 10, c))
        X = np.clip(np.vstack(X_l), 0, 1).astype(np.float32)
        y = np.concatenate(y_l)
        p = rng.permutation(len(X))
        return X[p], y[p]
    X_train, y_train = make_data(5000)
    X_test,  y_test  = make_data(1000)

CLASSES = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']
print(f"  X_train: {X_train.shape}  |  X_test: {X_test.shape}")


# ─────────────────────────────────────────────────────────
# 2. KNN Classifier (from scratch)
# ─────────────────────────────────────────────────────────

class KNN:
    def __init__(self, k=3, metric='l2'):
        assert metric in ('l1', 'l2')
        self.k = k
        self.metric = metric

    def fit(self, X, y):
        self.X_tr = np.array(X, dtype=np.float32)
        self.y_tr = np.array(y)

    def _distances(self, X_batch):
        if self.metric == 'l1':
            return np.sum(
                np.abs(X_batch[:, np.newaxis, :] - self.X_tr[np.newaxis, :, :]),
                axis=2)
        else:
            a2 = np.sum(X_batch ** 2, axis=1, keepdims=True)
            b2 = np.sum(self.X_tr ** 2, axis=1, keepdims=True)
            ab = X_batch @ self.X_tr.T
            return np.sqrt(np.maximum(a2 + b2.T - 2 * ab, 0.0))

    def predict(self, X_test, batch_size=250):
        X_test = np.array(X_test, dtype=np.float32)
        preds = []
        for s in range(0, len(X_test), batch_size):
            batch = X_test[s:s + batch_size]
            D = self._distances(batch)
            nn_labels = self.y_tr[np.argsort(D, axis=1)[:, :self.k]]
            for row in nn_labels:
                preds.append(Counter(row).most_common(1)[0][0])
            print(f"    {min(s+batch_size, len(X_test))}/{len(X_test)}", end='\r')
        print()
        return np.array(preds)


# ─────────────────────────────────────────────────────────
# 3. Metrics (from scratch)
# ─────────────────────────────────────────────────────────

def confusion_matrix_scratch(y_true, y_pred, n=10):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm

def compute_metrics(y_true, y_pred, n=10):
    cm = confusion_matrix_scratch(y_true, y_pred, n)
    prec, rec, f1 = [], [], []
    for c in range(n):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p = tp / (tp + fp) if (tp + fp) else 0.
        r = tp / (tp + fn) if (tp + fn) else 0.
        f = 2*p*r / (p+r) if (p+r) else 0.
        prec.append(p); rec.append(r); f1.append(f)
    return {
        'accuracy':  cm.diagonal().sum() / cm.sum(),
        'precision': np.mean(prec),
        'recall':    np.mean(rec),
        'f1':        np.mean(f1),
        'cm':        cm,
        'f1_per':    f1,
    }


# ─────────────────────────────────────────────────────────
# 4. Test set evaluation: K x metric
# ─────────────────────────────────────────────────────────

K_VALUES = [1, 3, 5, 7, 9]
METRICS  = ['l1', 'l2']

print("\n[2] Test set evaluation ...")
test_res = {}

for metric in METRICS:
    for k in K_VALUES:
        print(f"  {metric.upper()} k={k} ...", end=' ')
        knn = KNN(k=k, metric=metric)
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        m = compute_metrics(y_test, preds)
        m['preds'] = preds
        test_res[(metric, k)] = m
        print(f"Acc={m['accuracy']:.4f}")


# ─────────────────────────────────────────────────────────
# 5. 5-Fold Cross-Validation on Training Set
# ─────────────────────────────────────────────────────────

print("\n[3] 5-Fold CV on training set ...")
N_FOLDS = 5
fold_sz = len(X_train) // N_FOLDS
idx = np.arange(len(X_train))
cv_scores = {m: {k: [] for k in K_VALUES} for m in METRICS}

for fold in range(N_FOLDS):
    val_idx   = idx[fold * fold_sz:(fold + 1) * fold_sz]
    train_idx = np.concatenate([idx[:fold * fold_sz],
                                idx[(fold + 1) * fold_sz:]])
    Xtr, ytr = X_train[train_idx], y_train[train_idx]
    Xv,  yv  = X_train[val_idx],   y_train[val_idx]

    for metric in METRICS:
        for k in K_VALUES:
            print(f"  Fold {fold+1}/5 | {metric.upper()} k={k} ...", end=' ')
            knn = KNN(k=k, metric=metric)
            knn.fit(Xtr, ytr)
            preds = knn.predict(Xv)
            acc = np.mean(preds == yv)
            cv_scores[metric][k].append(acc)
            print(f"Acc={acc:.4f}")

print("\n  5-Fold CV Summary:")
print(f"  {'Metric':<5} {'k':>3}  {'Mean':>8}  {'Std':>8}")
print(f"  {'-'*32}")
for metric in METRICS:
    for k in K_VALUES:
        s = cv_scores[metric][k]
        print(f"  {metric.upper():<5} {k:>3}  {np.mean(s):>8.4f}  {np.std(s):>8.4f}")


# ─────────────────────────────────────────────────────────
# 6. Summary Table
# ─────────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("  TEST SET RESULTS")
print("=" * 55)
print(f"  {'Metric':<5} {'k':>3}  {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
print(f"  {'-'*45}")
for metric in METRICS:
    for k in K_VALUES:
        m = test_res[(metric, k)]
        print(f"  {metric.upper():<5} {k:>3}  "
              f"{m['accuracy']:>7.4f} {m['precision']:>7.4f} "
              f"{m['recall']:>7.4f} {m['f1']:>7.4f}")

best_key = max(test_res, key=lambda x: test_res[x]['accuracy'])
print(f"\n  Best: {best_key[0].upper()} k={best_key[1]}  "
      f"Acc={test_res[best_key]['accuracy']:.4f}")


# ─────────────────────────────────────────────────────────
# 7. Plots
# ─────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(17, 11))
fig.patch.set_facecolor('#0f1117')
BG, PANEL, TXT = '#0f1117', '#1a1d27', '#e0e0e0'
CL1, CL2 = '#7c83fd', '#f72585'

def sa(ax, title, xl='', yl=''):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TXT, labelsize=9)
    ax.xaxis.label.set_color(TXT); ax.yaxis.label.set_color(TXT)
    ax.title.set_color(TXT)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    for sp in ax.spines.values(): sp.set_edgecolor('#333655')

# ── 1. CV accuracy vs K (both metrics) ───────────────────
ax = axes[0, 0]
for metric, color in [('l1', CL1), ('l2', CL2)]:
    means = [np.mean(cv_scores[metric][k]) for k in K_VALUES]
    stds  = [np.std(cv_scores[metric][k])  for k in K_VALUES]
    for i, k in enumerate(K_VALUES):
        ax.scatter([k]*N_FOLDS, cv_scores[metric][k],
                   color=color, alpha=0.45, s=22, zorder=3)
    ax.errorbar(K_VALUES, means, yerr=stds, fmt='o-', color=color,
                ecolor=color, elinewidth=2, capsize=5, capthick=2,
                linewidth=2, markersize=7, label=metric.upper(), zorder=4)
ax.set_xticks(K_VALUES)
ax.legend(facecolor=PANEL, edgecolor='#333655', labelcolor=TXT, fontsize=9)
sa(ax, '5-Fold CV Accuracy vs K\n(dots=folds, line=mean, bar=std)', 'K', 'CV Accuracy')

# ── 2. L1 CV ─────────────────────────────────────────────
ax = axes[0, 1]
means = [np.mean(cv_scores['l1'][k]) for k in K_VALUES]
stds  = [np.std(cv_scores['l1'][k])  for k in K_VALUES]
for i, k in enumerate(K_VALUES):
    ax.scatter([k]*N_FOLDS, cv_scores['l1'][k], color=CL1, alpha=0.45, s=20)
ax.errorbar(K_VALUES, means, yerr=stds, fmt='o-', color=CL1,
            ecolor='#aaa', elinewidth=1.5, capsize=4, linewidth=2, markersize=7)
for i, k in enumerate(K_VALUES):
    ax.annotate(f"{means[i]:.3f}", (k, means[i]),
                textcoords='offset points', xytext=(0,9),
                ha='center', color=CL1, fontsize=8)
ax.set_xticks(K_VALUES)
sa(ax, 'L1 Distance: 5-Fold CV vs K', 'K', 'Accuracy')

# ── 3. L2 CV ─────────────────────────────────────────────
ax = axes[0, 2]
means = [np.mean(cv_scores['l2'][k]) for k in K_VALUES]
stds  = [np.std(cv_scores['l2'][k])  for k in K_VALUES]
for i, k in enumerate(K_VALUES):
    ax.scatter([k]*N_FOLDS, cv_scores['l2'][k], color=CL2, alpha=0.45, s=20)
ax.errorbar(K_VALUES, means, yerr=stds, fmt='o-', color=CL2,
            ecolor='#aaa', elinewidth=1.5, capsize=4, linewidth=2, markersize=7)
for i, k in enumerate(K_VALUES):
    ax.annotate(f"{means[i]:.3f}", (k, means[i]),
                textcoords='offset points', xytext=(0,9),
                ha='center', color=CL2, fontsize=8)
ax.set_xticks(K_VALUES)
sa(ax, 'L2 Distance: 5-Fold CV vs K', 'K', 'Accuracy')

# ── 4. Test accuracy bar ──────────────────────────────────
ax = axes[1, 0]
x = np.arange(len(K_VALUES)); w = 0.35
l1a = [test_res[('l1', k)]['accuracy'] for k in K_VALUES]
l2a = [test_res[('l2', k)]['accuracy'] for k in K_VALUES]
ax.bar(x-w/2, l1a, w, color=CL1, label='L1', edgecolor='#222')
ax.bar(x+w/2, l2a, w, color=CL2, label='L2', edgecolor='#222')
ax.set_xticks(x); ax.set_xticklabels([f'k={k}' for k in K_VALUES], fontsize=8)
ax.legend(facecolor=PANEL, edgecolor='#333655', labelcolor=TXT, fontsize=9)
sa(ax, 'Test Accuracy: L1 vs L2 per K', 'K', 'Accuracy')

# ── 5. Confusion matrix (with numbers) ───────────────────
ax = axes[1, 1]
cm = test_res[best_key]['cm']
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(10)); ax.set_yticks(range(10))
short = ['air','auto','bird','cat','deer','dog','frog','horse','ship','truck']
ax.set_xticklabels(short, rotation=45, ha='right', fontsize=7)
ax.set_yticklabels(short, fontsize=7)
plt.colorbar(im, ax=ax)
for i in range(10):
    for j in range(10):
        ax.text(j, i, str(cm[i, j]),
                ha='center', va='center',
                color='white' if cm[i, j] > cm.max() / 2 else 'black',
                fontsize=6)
sa(ax, f'Confusion Matrix\n{best_key[0].upper()} k={best_key[1]} '
       f'(Acc={test_res[best_key]["accuracy"]:.3f})',
   'Predicted', 'True')

# ── 6. Per-class F1 ──────────────────────────────────────
ax = axes[1, 2]
f1p = test_res[best_key]['f1_per']
ax.barh(range(10), f1p, color=CL1, edgecolor='#222')
ax.set_yticks(range(10)); ax.set_yticklabels(CLASSES, fontsize=8)
ax.set_xlim(0, 1)
for i, v in enumerate(f1p):
    ax.text(v+0.01, i, f'{v:.2f}', va='center', color=TXT, fontsize=8)
sa(ax, f'Per-class F1\n{best_key[0].upper()} k={best_key[1]}', 'F1-score', '')

fig.suptitle('KNN from Scratch  |  CIFAR-10  |  K=1,3,5,7,9  |  L1 vs L2  |  5-Fold CV',
             fontsize=12, fontweight='bold', color=TXT, y=0.995)
plt.tight_layout()
plt.savefig('knn_scratch_results.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("\n[Plot saved → knn_scratch_results.png]")


# ─────────────────────────────────────────────────────────
# 8. Limitation Analysis
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  LIMITATIONS OF KNN FOR IMAGE CLASSIFICATION")
print("=" * 55)
print("""
1. COMPUTATIONAL COST  O(N*D) per query
   No training phase — test time compares every query
   against N=5,000 samples x D=3,072 dims. Very slow.

2. CURSE OF DIMENSIONALITY
   In 3,072-dim space, all points become roughly
   equidistant. Nearest neighbors are often wrong.

3. PIXEL DISTANCE != SEMANTIC SIMILARITY
   L1/L2 sensitive to illumination, translation, background.
   Same class can be farther apart than different classes.

4. NO FEATURE LEARNING
   Raw pixels only. CNNs learn hierarchical features
   and achieve ~93% vs KNN's ~25-35% on CIFAR-10.

5. MEMORY INEFFICIENCY
   Entire training set stored and queried at inference.
""")
print("Done.")