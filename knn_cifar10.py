import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("  KNN Classification (Final Stable Version)")
print("=" * 60)

np.random.seed(42)

# =====================================================
# 1. 데이터 로드 (CIFAR 실패 → 자동 대체)
# =====================================================
print("\n[1] Loading dataset...")

use_cifar = False

try:
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.ToTensor()
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)

    X_raw = train_set.data
    y_raw = np.array(train_set.targets)

    idx = np.random.choice(len(X_raw), 3000, replace=False)
    X = X_raw[idx].reshape(3000, -1) / 255.0
    y = y_raw[idx]

    print("  CIFAR-10 loaded successfully")
    use_cifar = True

except:
    print("  CIFAR-10 download failed → using synthetic dataset")

    # ✔ 핵심: 겹치는 데이터 생성
    X, y = make_classification(
        n_samples=2000,
        n_features=50,
        n_informative=30,
        n_redundant=10,
        n_classes=10,
        n_clusters_per_class=2,
        class_sep=0.5,   # ⭐ 중요 (그래프 살리는 핵심)
        random_state=42
    )

# =====================================================
# 2. 전처리
# =====================================================
scaler = StandardScaler()
X = scaler.fit_transform(X)

print(f"  Dataset shape: {X.shape}")

# =====================================================
# 3. Part A: Train/Test Split
# =====================================================
print("\n[2] Part A: Train/Test Split")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred, average='macro'):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred, average='macro'):.4f}")
print(f"  F1-score:  {f1_score(y_test, y_pred, average='macro'):.4f}")

# =====================================================
# 4. Part B: Train/Validation/Test
# =====================================================
print("\n[3] Part B: Train/Validation/Test")

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.2, stratify=y_trainval, random_state=42)

k_candidates = list(range(1, 21))
val_scores = []

for k in k_candidates:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn.fit(X_train, y_train)
    val_acc = accuracy_score(y_val, knn.predict(X_val))
    val_scores.append(val_acc)

best_k = k_candidates[np.argmax(val_scores)]
print(f"  Best k (validation): {best_k}")

# 테스트 평가
knn = KNeighborsClassifier(n_neighbors=best_k, weights='distance')
knn.fit(X_trainval, y_trainval)
y_pred = knn.predict(X_test)

print(f"  Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# =====================================================
# 5. Part C: 5-Fold Cross Validation
# =====================================================
print("\n[4] Part C: 5-Fold Cross Validation")

k_range = list(range(1, 31))
cv_mean = []
cv_std = []

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    scores = cross_val_score(knn, X, y, cv=skf, scoring='accuracy')
    cv_mean.append(scores.mean())
    cv_std.append(scores.std())

best_k_cv = k_range[np.argmax(cv_mean)]
print(f"  Best k (CV): {best_k_cv}")

# =====================================================
# 6. 그래프 출력
# =====================================================
plt.figure(figsize=(8,5))
plt.errorbar(k_range, cv_mean, yerr=cv_std, fmt='o-')
plt.xlabel("k (Number of Neighbors)")
plt.ylabel("Accuracy")
plt.title("5-Fold Cross-Validation Accuracy vs k")
plt.grid(True)

plt.savefig("knn_result.png")
plt.show()

print("\n[Done] Graph saved as knn_result.png")
