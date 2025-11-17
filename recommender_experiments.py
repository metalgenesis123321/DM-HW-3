import pandas as pd, numpy as np, os, argparse, math
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity

def pmf_train_predict(train_df, test_df, n_users, n_items, n_factors=30, lr=0.01, reg=0.05, n_epochs=25, seed=0):
    rng = np.random.RandomState(seed)
    P = 0.1*rng.randn(n_users, n_factors)
    Q = 0.1*rng.randn(n_items, n_factors)
    bu = np.zeros(n_users); bi = np.zeros(n_items)
    mu = train_df['rating'].mean()
    rows = train_df[['uidx','iidx','rating']].values.copy()
    for epoch in range(n_epochs):
        np.random.shuffle(rows)
        for u,i,r in rows:
            u=int(u); i=int(i)
            pred = mu + bu[u] + bi[i] + P[u].dot(Q[i])
            e = r - pred
            bu[u] += lr * (e - reg * bu[u])
            bi[i] += lr * (e - reg * bi[i])
            Pu = P[u].copy(); Qi = Q[i].copy()
            P[u] += lr * (e * Qi - reg * Pu)
            Q[i] += lr * (e * Pu - reg * Qi)
    preds=[]
    trues=[]
    for _,row in test_df.iterrows():
        u=int(row['uidx']); i=int(row['iidx']); r=row['rating']
        pred = mu + bu[u] + bi[i] + P[u].dot(Q[i])
        preds.append(pred); trues.append(r)
    return trues, preds

def compute_sims(R, metric='cosine'):
    if metric == 'cosine':
        R0 = np.nan_to_num(R, nan=0.0)
        return cosine_similarity(R0)
    n = R.shape[0]; sims = np.zeros((n,n))
    if metric == 'msd':
        for u in range(n):
            for v in range(n):
                mask = ~np.isnan(R[u]) & ~np.isnan(R[v])
                if mask.sum()==0:
                    sims[u,v]=0.0
                else:
                    sims[u,v] = 1.0 / (1.0 + ((R[u,mask]-R[v,mask])**2).mean())
        return sims
    if metric == 'pearson':
        for u in range(n):
            for v in range(n):
                mask = ~np.isnan(R[u]) & ~np.isnan(R[v])
                if mask.sum() < 2:
                    sims[u,v]=0.0
                else:
                    a = R[u,mask] - R[u,mask].mean()
                    b = R[v,mask] - R[v,mask].mean()
                    denom = (np.linalg.norm(a) * np.linalg.norm(b))
                    sims[u,v] = (a.dot(b) / denom) if denom>0 else 0.0
        return sims
    raise ValueError("Unknown metric")

def predict_usercf(u, i, sims, R, k=20):
    raters = np.where(~np.isnan(R[:,i]))[0]
    if len(raters)==0:
        return np.nanmean(R[~np.isnan(R)])
    user_sims = sims[u, raters]
    idx = np.argsort(-user_sims)[:k]
    top = raters[idx]; tops = user_sims[idx]
    mask = tops != 0
    if mask.sum()==0:
        return np.nanmean(R[~np.isnan(R)])
    return (tops[mask] * R[top[mask], i]).sum() / (np.abs(tops[mask]).sum() + 1e-12)

def predict_itemcf(u, i, sims_item, R, k=20):
    user_rated = np.where(~np.isnan(R[u]))[0]
    if len(user_rated)==0:
        return np.nanmean(R[~np.isnan(R)])
    sim_vals = sims_item[i, user_rated]
    idx = np.argsort(-sim_vals)[:k]; top = user_rated[idx]; tops = sim_vals[idx]
    mask = tops != 0
    if mask.sum()==0:
        return np.nanmean(R[~np.isnan(R)])
    return (tops[mask] * R[u, top[mask]]).sum() / (np.abs(tops[mask]).sum() + 1e-12)

def main(ratings_csv, outdir, sample=0, folds=5):
    os.makedirs(outdir, exist_ok=True)
    ratings = pd.read_csv(ratings_csv)
    if sample and sample < len(ratings):
        ratings = ratings.sample(n=sample, random_state=0).reset_index(drop=True)
    user_ids = ratings['userId'].unique()
    item_ids = ratings['movieId'].unique()
    u2i = {u:i for i,u in enumerate(user_ids)}
    m2i = {m:i for i,m in enumerate(item_ids)}
    ratings['uidx'] = ratings['userId'].map(u2i); ratings['iidx'] = ratings['movieId'].map(m2i)
    n_users = len(user_ids); n_items = len(item_ids)
    print("n_users,n_items:", n_users, n_items)
    kf = KFold(n_splits=folds, shuffle=True, random_state=0)
    pmf_maes=[]; pmf_rmses=[]
    usercf_summary = {'cosine':[], 'msd':[], 'pearson':[]}
    itemcf_summary = {'cosine':[], 'msd':[], 'pearson':[]}
    for fold, (train_idx, test_idx) in enumerate(kf.split(ratings), 1):
        train_df = ratings.iloc[train_idx]; test_df = ratings.iloc[test_idx]
        tr, pr = pmf_train_predict(train_df, test_df, n_users, n_items, n_factors=30, lr=0.01, reg=0.05, n_epochs=15, seed=fold)
        pmf_maes.append(mean_absolute_error(tr, pr)); pmf_rmses.append(math.sqrt(mean_squared_error(tr, pr)))
        Rtrain = np.full((n_users, n_items), np.nan)
        for _,r in train_df.iterrows():
            Rtrain[int(r['uidx']), int(r['iidx'])] = r['rating']
        for metric in ['cosine','msd','pearson']:
            sims_user = compute_sims(Rtrain, metric=metric)
            preds_u=[]; tr_u=[]
            for _,row in test_df.iterrows():
                preds_u.append(predict_usercf(int(row['uidx']), int(row['iidx']), sims_user, Rtrain, k=20))
                tr_u.append(row['rating'])
            usercf_summary[metric].append((mean_absolute_error(tr_u, preds_u), math.sqrt(mean_squared_error(tr_u, preds_u))))
            SimsItem = compute_sims(Rtrain.T, metric=metric)
            preds_i=[]; tr_i=[]
            for _,row in test_df.iterrows():
                preds_i.append(predict_itemcf(int(row['uidx']), int(row['iidx']), SimsItem, Rtrain, k=20))
                tr_i.append(row['rating'])
            itemcf_summary[metric].append((mean_absolute_error(tr_i, preds_i), math.sqrt(mean_squared_error(tr_i, preds_i))))
    pmf_summary = (np.mean(pmf_maes), np.std(pmf_maes), np.mean(pmf_rmses), np.std(pmf_rmses))
    usercf_stats = {m:(np.mean([v[0] for v in vals]), np.std([v[0] for v in vals]), np.mean([v[1] for v in vals]), np.std([v[1] for v in vals])) for m,vals in usercf_summary.items()}
    itemcf_stats = {m:(np.mean([v[0] for v in vals]), np.std([v[0] for v in vals]), np.mean([v[1] for v in vals]), np.std([v[1] for v in vals])) for m,vals in itemcf_summary.items()}
    with open(os.path.join(outdir,'recommender_summary.txt'),'w') as f:
        f.write("pmf_summary: %s\n" % str(pmf_summary))
        f.write("usercf: %s\n" % str(usercf_stats))
        f.write("itemcf: %s\n" % str(itemcf_stats))
    print("Saved results to", outdir)
    print("PMF summary:", pmf_summary)
    print("UserCF:", usercf_stats)
    print("ItemCF:", itemcf_stats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratings', default='ratings_small.csv')
    parser.add_argument('--out', default='results_reco')
    parser.add_argument('--sample', type=int, default=0, help='sample number of ratings for fast runs (0 = full)')
    parser.add_argument('--folds', type=int, default=5)
    args = parser.parse_args()
    main(args.ratings, args.out, sample=args.sample or 0, folds=args.folds)
