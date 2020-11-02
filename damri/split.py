from copy import deepcopy

from dpipe.split import train_val_test_split, train_test_split

def one2all(df, val_size=2, seed=0xBadCafe):
    """Train on 1 domain, test on (n - 1) domains."""
    folds = sorted(df.fold.unique())
    split = []
    for f in folds:
        idx_b = df[df.fold == f].index.tolist()
        test_ids = df[df.fold != f].index.tolist()
        train_ids, val_ids = train_test_split(idx_b, test_size=val_size, random_state=seed)
        split.append([train_ids, val_ids, test_ids])
    return split


def all2one(df, val_size=2, seed=0xBadCafe):
    """Train on (n - 1) domains, test on 1 domains."""
    folds = sorted(df.fold.unique())
    split = []
    for f in folds:
        idx_b = df[df.fold != f].index.tolist()
        test_ids = df[df.fold == f].index.tolist()
        train_ids, val_ids = train_test_split(idx_b, test_size=val_size, random_state=seed)
        split.append([train_ids, val_ids, test_ids])
    return split


def single_cv(df, n_splits=3, val_size=2, seed=0xBadCafe):
    """Cross-validation inside every domain."""
    folds = sorted(df.fold.unique())
    split = []
    for f in folds:
        idx_b = df[df.fold == f].index.tolist()
        cv_b = train_val_test_split(idx_b, val_size=val_size, n_splits=n_splits, random_state=seed)
        for cv in cv_b:
            split.append(cv)
    return split


def one2one(df, val_size=2, n_add_ids=5, train_on_add_only=False, seed=0xBadCafe):
    folds = sorted(df.fold.unique())
    split = []
    for fb in folds:
        folds_o = set(folds) - {fb}
        ids_train, ids_val = train_test_split(df[df.fold == fb].index.tolist(), test_size=val_size, random_state=seed)
        if train_on_add_only:
            ids_train = []

        for fo in folds_o:
            ids_test, ids_train_add = train_test_split(df[df.fold == fo].index.tolist(), test_size=n_add_ids,
                                                       random_state=seed)
            split.append([deepcopy(ids_train) + ids_train_add, ids_val, ids_test])
    return split


# ======================================= Deprecated =======================================


def one2one_cv(df, n_splits=3, val_size=2, n_add_ids=10, train_on_add_only=False, seed=0xBadCafe):
    folds = sorted(df.fold.unique())

    split = []
    for f_b in folds:
        folds_o = list(set(folds) - {f_b})
        idx_b = df[df.fold == f_b].index.tolist()
        cv3 = train_val_test_split(idx_b, val_size=val_size, n_splits=n_splits, random_state=seed)

        for cv3_iter, f_o in zip([cv3] * len(folds_o), folds_o):
            idx_o_test, idx_o_train = train_test_split(df[df.fold == f_o].index.tolist(),
                                                       test_size=n_add_ids, random_state=seed)
            for cv in deepcopy(cv3_iter):
                if train_on_add_only:
                    cv[0] = idx_o_train
                else:
                    cv[0] += idx_o_train
                cv[2] += idx_o_test
                split.append(cv)

    return split


def one2one_zipped_cv(df, n_splits=3, val_size=2, train_on_add_only=False, seed=0xBadCafe):
    folds = sorted(df.fold.unique())

    split = []
    for f_b in folds:
        folds_o = list(set(folds) - {f_b})
        idx_b = df[df.fold == f_b].index.tolist()
        cv3 = train_val_test_split(idx_b, val_size=val_size, n_splits=n_splits, random_state=seed)

        for cv3_iter, f_o in zip([cv3] * len(folds_o), folds_o):
            idx_o = df[df.fold == f_o].index.tolist()
            cv3_o = train_val_test_split(idx_o, val_size=val_size, n_splits=n_splits, random_state=seed)

            for cv_b, cv_o in zip(deepcopy(cv3_iter), cv3_o):
                for i in range(len(cv_b)):
                    if (i == 0) and train_on_add_only:
                        cv_b[i] = cv_o[i]
                    else:
                        cv_b[i] += cv_o[i]
                split.append(cv_b)

    return split


def one2all_cv(df, n_splits=3, val_size=2, seed=0xBadCafe):
    folds = sorted(df.fold.unique())
    split = []
    for f in folds:
        idx_b = df[df.fold == f].index.tolist()
        idx_o = df[df.fold != f].index.tolist()
        cv3 = train_val_test_split(idx_b, val_size=val_size, n_splits=n_splits, random_state=seed)
        for cv in cv3:
            cv[-1] += idx_o
            split.append(cv)
    return split
