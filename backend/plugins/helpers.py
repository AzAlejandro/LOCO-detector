from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

import cv2
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ..features import build_features


def _balance_indices(y: np.ndarray, rng_seed: int = 42, max_per_class: int = 25000) -> np.ndarray:
    y = np.asarray(y).reshape(-1)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    if idx_pos.size == 0 or idx_neg.size == 0:
        raise ValueError('Se requieren muestras FG y BG.')
    min_class = int(min(idx_pos.size, idx_neg.size))
    max_ratio = 4
    n_per = int(min(max(idx_pos.size, idx_neg.size), max(min_class, min_class * max_ratio), max_per_class))
    rng = np.random.RandomState(rng_seed)
    sel_pos = rng.choice(idx_pos, size=n_per, replace=idx_pos.size < n_per)
    sel_neg = rng.choice(idx_neg, size=n_per, replace=idx_neg.size < n_per)
    sel = np.concatenate([sel_pos, sel_neg])
    rng.shuffle(sel)
    return sel


def _balance_indices_by_class(y: np.ndarray, rng_seed: int = 42, max_per_class: int = 25000) -> np.ndarray:
    y = np.asarray(y).reshape(-1)
    classes, counts = np.unique(y, return_counts=True)
    if classes.size < 2:
        raise ValueError('Se requieren al menos dos clases de scribbles.')
    target = int(min(max_per_class, max(int(np.max(counts)), 1)))
    rng = np.random.RandomState(rng_seed)
    selected: list[np.ndarray] = []
    for cls in classes:
        idx = np.where(y == cls)[0]
        if idx.size == 0:
            continue
        selected.append(rng.choice(idx, size=target, replace=idx.size < target))
    if not selected:
        raise ValueError('No hay muestras de scribbles balanceables.')
    sel = np.concatenate(selected)
    rng.shuffle(sel)
    return sel


def _fit_binary_classifier(kind: str, x: np.ndarray, y: np.ndarray, n_estimators: int, class_weight: str | None = None, n_classes: int = 2):
    kind = str(kind).lower().strip()
    if kind == 'rf':
        clf = RandomForestClassifier(
            n_estimators=int(max(20, n_estimators)),
            max_depth=16,
            min_samples_leaf=2,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=42,
        )
        setattr(clf, '_sr_impl', 'random_forest')
    elif kind == 'extratrees':
        clf = ExtraTreesClassifier(
            n_estimators=int(max(40, n_estimators)),
            max_depth=None,
            min_samples_leaf=1,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=42,
        )
        setattr(clf, '_sr_impl', 'extra_trees')
    elif kind == 'xgboost':
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            clf = GradientBoostingClassifier(random_state=42)
            setattr(clf, '_sr_fallback', 'xgboost_unavailable_gradient_boosting')
            setattr(clf, '_sr_fallback_detail', str(exc))
            setattr(clf, '_sr_impl', 'gradient_boosting_fallback')
        else:
            xgb_kwargs = {
                'n_estimators': int(max(40, n_estimators)),
                'max_depth': 6,
                'learning_rate': 0.08,
                'subsample': 0.85,
                'colsample_bytree': 0.85,
                'n_jobs': 4,
                'random_state': 42,
                'reg_lambda': 1.0,
            }
            if int(n_classes) > 2:
                xgb_kwargs.update({'objective': 'multi:softprob', 'eval_metric': 'mlogloss', 'num_class': int(n_classes)})
            else:
                xgb_kwargs.update({'objective': 'binary:logistic', 'eval_metric': 'logloss'})
            clf = XGBClassifier(**xgb_kwargs)
            setattr(clf, '_sr_impl', 'xgboost')
            try:
                setattr(clf, '_sr_package_version', version('xgboost'))
            except PackageNotFoundError:
                try:
                    setattr(clf, '_sr_package_version', version('xgboost-cpu'))
                except PackageNotFoundError:
                    pass
    elif kind == 'catboost':
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            clf = GradientBoostingClassifier(random_state=42)
            setattr(clf, '_sr_fallback', 'catboost_unavailable_gradient_boosting')
            setattr(clf, '_sr_fallback_detail', str(exc))
            setattr(clf, '_sr_impl', 'gradient_boosting_fallback')
        else:
            clf = CatBoostClassifier(
                iterations=int(max(40, n_estimators)),
                depth=6,
                learning_rate=0.08,
                loss_function='MultiClass' if int(n_classes) > 2 else 'Logloss',
                random_seed=42,
                verbose=False,
            )
            setattr(clf, '_sr_impl', 'catboost')
            try:
                setattr(clf, '_sr_package_version', version('catboost'))
            except PackageNotFoundError:
                pass
    elif kind == 'logreg':
        clf = LogisticRegression(max_iter=350, n_jobs=1, solver='lbfgs')
        setattr(clf, '_sr_impl', 'logistic_regression')
    else:
        raise ValueError(f'Clasificador no soportado: {kind}')

    clf.fit(x, y)
    return clf


def tree_pixel_prior(
    image_rgb: np.ndarray,
    labels: np.ndarray,
    *,
    classifier: str,
    n_estimators: int = 120,
    feature_variant: str = 'base',
    class_weight: str | None = None,
    strong_balance: bool = False,
    return_class_maps: bool = False,
) -> tuple[np.ndarray, dict[str, Any]] | tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    feats = build_features(image_rgb, variant=feature_variant)
    h, w, c = feats.shape
    x_all = feats.reshape(-1, c)

    lab = np.asarray(labels, dtype=np.uint8)
    mask = lab > 0
    if int(np.sum(mask)) < 2:
        raise ValueError('No hay suficientes scribbles.')

    x_train = feats[mask]
    y_raw = lab[mask].astype(np.uint8)
    has_fiber = bool(np.any(y_raw == 1))
    has_background = bool(np.any(y_raw == 3))
    has_negative = has_background or bool(np.any(y_raw == 2))
    if not has_fiber or not has_negative:
        raise ValueError('Se requieren muestras de fibra y una clase negativa/background.')

    classes = np.asarray(sorted(int(x) for x in np.unique(y_raw)), dtype=np.uint8)
    class_to_idx = {int(cls): idx for idx, cls in enumerate(classes.tolist())}
    y_train = np.asarray([class_to_idx[int(v)] for v in y_raw], dtype=np.uint8)
    if classes.size <= 2:
        sel = _balance_indices(y_train, max_per_class=30000 if strong_balance else 20000)
    else:
        sel = _balance_indices_by_class(y_train, max_per_class=30000 if strong_balance else 20000)
    x_bal = x_train[sel]
    y_bal = y_train[sel]

    clf = _fit_binary_classifier(
        classifier,
        x_bal,
        y_bal,
        n_estimators=n_estimators,
        class_weight=class_weight,
        n_classes=int(classes.size),
    )
    proba = clf.predict_proba(x_all)
    if proba.shape[1] == 1:
        pred = np.asarray(clf.predict(x_all)).reshape(-1)
        fiber_idx = class_to_idx.get(1, 0)
        p_fg = (pred == fiber_idx).astype(np.float32)
    else:
        fiber_idx = class_to_idx.get(1, 0)
        p_fg = proba[:, fiber_idx].astype(np.float32)

    prior = p_fg.reshape(h, w)
    prob_maps: dict[str, np.ndarray] = {'fiber_prob': prior}
    for label_id, name in ((2, 'halo_prob'), (3, 'background_prob')):
        idx = class_to_idx.get(label_id)
        if idx is None or proba.ndim != 2 or idx >= proba.shape[1]:
            prob_maps[name] = np.zeros((h, w), dtype=np.float32)
        else:
            prob_maps[name] = proba[:, idx].astype(np.float32).reshape(h, w)
    meta: dict[str, Any] = {
        'classifier': classifier,
        'classifier_impl': str(getattr(clf, '_sr_impl', classifier)),
        'classifier_package_version': str(getattr(clf, '_sr_package_version', '')),
        'feature_profile': feature_variant,
        'n_estimators': int(n_estimators),
        'class_weight': class_weight,
        'train_samples': int(x_bal.shape[0]),
        'class_mode': 'multiclass_halo' if (2 in classes.tolist()) else 'multiclass_compat',
        'class_labels': [int(x) for x in classes.tolist()],
        'class_prob_maps_saved': bool(return_class_maps),
        'fg_marked_px': int(np.sum(lab == 1)),
        'halo_marked_px': int(np.sum(lab == 2)),
        'bg_marked_px': int(np.sum(lab == 3)),
    }
    fallback = getattr(clf, '_sr_fallback', None)
    if fallback:
        meta['fallback_used'] = True
        meta['blocker_reason'] = str(fallback)
        detail = getattr(clf, '_sr_fallback_detail', '')
        if detail:
            meta['blocker_detail'] = str(detail)
    prior_out = np.clip(prior, 0.0, 1.0).astype(np.float32)
    prob_maps = {k: np.clip(v, 0.0, 1.0).astype(np.float32) for k, v in prob_maps.items()}
    if return_class_maps:
        return prior_out, meta, prob_maps
    return prior_out, meta


def threshold_and_clean(mask_prob: np.ndarray, thr: float = 0.5, closing_radius: int = 1, min_hole_area: int = 32, min_obj_area: int = 24) -> np.ndarray:
    # Modo investigacion "raw": mascara binaria directa desde prior sin morfologia.
    # Se conservan los parametros para compatibilidad de firma entre experimentos.
    p = np.asarray(mask_prob, dtype=np.float32)
    return (p >= float(thr)).astype(np.uint8)
