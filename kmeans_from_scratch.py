import pandas as pd, numpy as np, os, time, argparse
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from collections import Counter

def pairwise_euclidean_sq(X, C):
    X_norm = np.sum(X**2, axis=1).reshape(-1,1)
    C_norm = np.sum(C**2, axis=1).reshape(1,-1)
    return X_norm + C_norm - 2.0 * (X.dot(C.T))

def init_centroids_pp(X, K, random_state=0):
    rng = np.random.RandomState(random_state)
    n = X.shape[0]
    centroids = np.zeros((K, X.shape[1]))
    centroids[0] = X[rng.randint(n)]
    for i in range(1, K):
        dists = np.min(np.linalg.norm(X[:,None,:] - centroids[None,:i,:], axis=2), axis=1)
        probs = dists**2
        if probs.sum() <= 0:
            centroids[i] = X[rng.randint(n)]
        else:
            probs = probs / probs.sum()
            idx = rng.choice(n, p=probs)
            centroids[i] = X[idx]
    return centroids

def kmeans_euclidean_fast(X, K, max_iter=200, tol=1e-6, random_state=0):
    rng = np.random.RandomState(random_state)
    centroids = init_centroids_pp(X, K, random_state=random_state)
    history=[]
    for it in range(1, max_iter+1):
        Dsq = pairwise_euclidean_sq(X, centroids)
        labels = np.argmin(Dsq, axis=1)
        new_centroids = np.zeros_like(centroids)
        for k in range(K):
            pts = X[labels==k]
            if len(pts)==0:
                new_centroids[k] = X[rng.randint(X.shape[0])]
            else:
                new_centroids[k] = pts.mean(axis=0)
        sse = np.sum(np.min(Dsq, axis=1))
        history.append((it, sse))
        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift < tol:
            return centroids, labels, history, it, 'tol'
    return centroids, labels, history, max_iter, 'max'

def kmeans_distance_fast(X, K, metric='cosine', max_iter=200, tol=1e-6, random_state=0):
    rng = np.random.RandomState(random_state)
    n,d = X.shape
    centroids = X[rng.choice(n, K, replace=False)].copy()
    history=[]
    for it in range(1, max_iter+1):
        if metric == 'cosine':
            Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
            cn = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
            D = 1.0 - Xn.dot(cn.T)
        elif metric == 'jaccard':
            Xs = X - X.min() if X.min() < 0 else X
            Cs = centroids - centroids.min() if centroids.min() < 0 else centroids
            num = np.minimum(Xs[:,None,:], Cs[None,:,:]).sum(axis=2)
            den = np.maximum(Xs[:,None,:], Cs[None,:,:]).sum(axis=2) + 1e-12
            sim = num / den
            D = 1.0 - sim
        labels = np.argmin(D, axis=1)
        new_centroids = np.zeros_like(centroids)
        for k in range(K):
            pts = X[labels==k]
            if len(pts) == 0:
                new_centroids[k] = X[rng.randint(n)]
            else:
                new_centroids[k] = pts.mean(axis=0)
        sse = np.sum((D[np.arange(n), labels])**2)
        history.append((it, sse))
        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift < tol:
            return centroids, labels, history, it, 'tol'
    return centroids, labels, history, max_iter, 'max'

def majority_label_accuracy(y, labels):
    mapping = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            mapping[c] = -1
        else:
            mapping[c] = Counter(y[idx]).most_common(1)[0][0]
    ypred = np.array([mapping[l] for l in labels])
    return np.mean(ypred == y), mapping

def main(data_csv, label_csv, outdir, max_iter=100):
    os.makedirs(outdir, exist_ok=True)
    X = pd.read_csv(data_csv, header=None).values.astype(float)
    y = pd.read_csv(label_csv, header=None).values.ravel()
    K = len(np.unique(y))
    print("Loaded X:", X.shape, "K:", K)
    ce, le, he, it_e, stop_e = kmeans_euclidean_fast(X, K, max_iter=max_iter, random_state=0)
    acc_e, map_e = majority_label_accuracy(y, le)
    print("Euclid done iters", it_e, "acc", acc_e)
    svd = TruncatedSVD(n_components=50, random_state=0)
    X50 = svd.fit_transform(X)
    cc, lc, hc, it_c, stop_c = kmeans_distance_fast(X50, K, metric='cosine', max_iter=max_iter, random_state=0)
    cj, lj, hj, it_j, stop_j = kmeans_distance_fast(X50, K, metric='jaccard', max_iter=max_iter, random_state=0)
    acc_c, map_c = majority_label_accuracy(y, lc)
    acc_j, map_j = majority_label_accuracy(y, lj)
    summary = pd.DataFrame([
        {'method':'euclidean','iterations':it_e,'stopped':stop_e,'accuracy':acc_e,'history_len':len(he)},
        {'method':'cosine(reduced)','iterations':it_c,'stopped':stop_c,'accuracy':acc_c,'history_len':len(hc)},
        {'method':'jaccard(reduced)','iterations':it_j,'stopped':stop_j,'accuracy':acc_j,'history_len':len(hj)}
    ])
    summary.to_csv(os.path.join(outdir,'kmeans_summary.csv'), index=False)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(9,5))
    plt.plot([h[0] for h in he], [h[1] for h in he], label='euclidean')
    plt.plot([h[0] for h in hc], [h[1] for h in hc], label='cosine(reduced)')
    plt.plot([h[0] for h in hj], [h[1] for h in hj], label='jaccard(reduced)')
    plt.xlabel('iteration'); plt.ylabel('objective'); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(outdir,'kmeans_hist.png'), dpi=150)
    print("Saved outputs to", outdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data.csv')
    parser.add_argument('--label', default='label.csv')
    parser.add_argument('--out', default='results')
    parser.add_argument('--max_iter', type=int, default=100)
    args = parser.parse_args()
    main(args.data, args.label, args.out, max_iter=args.max_iter)
