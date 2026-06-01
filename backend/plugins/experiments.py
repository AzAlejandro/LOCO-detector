from __future__ import annotations

from time import perf_counter

from .base import ExperimentInfo, ExperimentOutput, RunContext
from .helpers import (
    threshold_and_clean,
    tree_pixel_prior,
    unet_small_patch_prior,
)


def _status_from_meta(meta: dict[str, object], default: str = 'success') -> str:
    if str(meta.get('status_level', '')).lower() in {'success', 'warning', 'error'}:
        return str(meta.get('status_level')).lower()
    if meta.get('blocker_reason'):
        return 'warning'
    if bool(meta.get('fallback_used')):
        return 'warning'
    return default


class ExtraTreesPixelExperiment:
    info = ExperimentInfo(
        experiment_id='extratrees_pixel',
        group='A',
        display_name='ExtraTrees pixel-wise',
        description='Baseline rapido con features multiescala.',
        default_params={'n_estimators': 160, 'threshold': 0.5},
        implementation_status='native',
    )

    def run(self, ctx: RunContext) -> ExperimentOutput:
        t0 = perf_counter()
        prior, meta, prob_maps = tree_pixel_prior(ctx.image_rgb, ctx.labels, classifier='extratrees', n_estimators=int(ctx.params.get('n_estimators', 160)), return_class_maps=True)
        mask = threshold_and_clean(prior, thr=float(ctx.params.get('threshold', 0.5)), closing_radius=1, min_hole_area=32)
        meta.update({'runtime_ms': (perf_counter() - t0) * 1000.0})
        return ExperimentOutput(prior_map=prior, mask=mask, meta=meta, status_level=_status_from_meta(meta), prob_maps=prob_maps)


class RandomForestPixelExperiment:
    info = ExperimentInfo(
        experiment_id='rf_pixel',
        group='A',
        display_name='RandomForest pixel-wise',
        description='Comparativo directo frente a ExtraTrees.',
        default_params={'n_estimators': 120, 'threshold': 0.5},
        implementation_status='native',
    )

    def run(self, ctx: RunContext) -> ExperimentOutput:
        t0 = perf_counter()
        prior, meta, prob_maps = tree_pixel_prior(ctx.image_rgb, ctx.labels, classifier='rf', n_estimators=int(ctx.params.get('n_estimators', 120)), return_class_maps=True)
        mask = threshold_and_clean(prior, thr=float(ctx.params.get('threshold', 0.5)), closing_radius=1, min_hole_area=32)
        meta.update({'runtime_ms': (perf_counter() - t0) * 1000.0})
        return ExperimentOutput(prior_map=prior, mask=mask, meta=meta, status_level=_status_from_meta(meta), prob_maps=prob_maps)


class XGBoostPixelExperiment:
    info = ExperimentInfo(
        experiment_id='xgboost_pixel',
        group='A',
        display_name='XGBoost pixel-wise',
        description='Boosting para zonas dificiles; fallback si xgboost no esta disponible.',
        default_params={'n_estimators': 180, 'threshold': 0.5},
        implementation_status='native',
        requirements_hint='xgboost',
    )

    def run(self, ctx: RunContext) -> ExperimentOutput:
        t0 = perf_counter()
        prior, meta, prob_maps = tree_pixel_prior(ctx.image_rgb, ctx.labels, classifier='xgboost', n_estimators=int(ctx.params.get('n_estimators', 180)), return_class_maps=True)
        mask = threshold_and_clean(prior, thr=float(ctx.params.get('threshold', 0.5)), closing_radius=1, min_hole_area=32)
        meta.update({'runtime_ms': (perf_counter() - t0) * 1000.0})
        return ExperimentOutput(prior_map=prior, mask=mask, meta=meta, status_level=_status_from_meta(meta), prob_maps=prob_maps)


class CatBoostPixelExperiment:
    info = ExperimentInfo(
        experiment_id='catboost_pixel',
        group='A',
        display_name='CatBoost pixel-wise',
        description='Boosting robusto para zonas dificiles; fallback si catboost no esta disponible.',
        default_params={'n_estimators': 220, 'threshold': 0.5},
        implementation_status='native',
        requirements_hint='catboost',
    )

    def run(self, ctx: RunContext) -> ExperimentOutput:
        t0 = perf_counter()
        prior, meta, prob_maps = tree_pixel_prior(ctx.image_rgb, ctx.labels, classifier='catboost', n_estimators=int(ctx.params.get('n_estimators', 220)), return_class_maps=True)
        mask = threshold_and_clean(prior, thr=float(ctx.params.get('threshold', 0.5)), closing_radius=1, min_hole_area=32)
        meta.update({'runtime_ms': (perf_counter() - t0) * 1000.0})
        return ExperimentOutput(prior_map=prior, mask=mask, meta=meta, status_level=_status_from_meta(meta), prob_maps=prob_maps)


class ExtraTreesBalancedExperiment:
    info = ExperimentInfo(
        experiment_id='extratrees_balanced',
        group='C',
        display_name='ExtraTrees balanceado',
        description='Balance de clases agresivo para mitigar dominancia de fondo.',
        default_params={'n_estimators': 180, 'threshold': 0.5},
        implementation_status='native',
    )

    def run(self, ctx: RunContext) -> ExperimentOutput:
        t0 = perf_counter()
        prior, meta, prob_maps = tree_pixel_prior(
            ctx.image_rgb,
            ctx.labels,
            classifier='extratrees',
            n_estimators=int(ctx.params.get('n_estimators', 180)),
            class_weight='balanced',
            strong_balance=True,
            return_class_maps=True,
        )
        mask = threshold_and_clean(prior, thr=float(ctx.params.get('threshold', 0.5)), closing_radius=1, min_hole_area=32)
        meta.update({'runtime_ms': (perf_counter() - t0) * 1000.0})
        return ExperimentOutput(prior_map=prior, mask=mask, meta=meta, status_level='success', prob_maps=prob_maps)


class ContextFeaturesExperiment:
    info = ExperimentInfo(
        experiment_id='context_features_variant',
        group='C',
        display_name='Clasificador con features de contexto',
        description='ExtraTrees con estadisticas locales ampliadas y contexto multiescala.',
        default_params={'n_estimators': 180, 'threshold': 0.5},
        implementation_status='native',
    )

    def run(self, ctx: RunContext) -> ExperimentOutput:
        t0 = perf_counter()
        prior, meta, prob_maps = tree_pixel_prior(
            ctx.image_rgb,
            ctx.labels,
            classifier='extratrees',
            n_estimators=int(ctx.params.get('n_estimators', 180)),
            feature_variant='context',
            return_class_maps=True,
        )
        mask = threshold_and_clean(prior, thr=float(ctx.params.get('threshold', 0.5)), closing_radius=1, min_hole_area=48)
        meta.update({'runtime_ms': (perf_counter() - t0) * 1000.0})
        return ExperimentOutput(prior_map=prior, mask=mask, meta=meta, status_level='success', prob_maps=prob_maps)


class ClassifierMorphMinExperiment:
    info = ExperimentInfo(
        experiment_id='classifier_morph_min',
        group='D',
        display_name='Clasificador + morfologia minima',
        description='Threshold + limpieza morfologica minima (opening/closing/hole-fill leve).',
        default_params={'n_estimators': 160, 'threshold': 0.5, 'closing_radius': 1},
        implementation_status='native',
    )

    def run(self, ctx: RunContext) -> ExperimentOutput:
        t0 = perf_counter()
        prior, meta, prob_maps = tree_pixel_prior(ctx.image_rgb, ctx.labels, classifier='extratrees', n_estimators=int(ctx.params.get('n_estimators', 160)), return_class_maps=True)
        mask = threshold_and_clean(
            prior,
            thr=float(ctx.params.get('threshold', 0.5)),
            closing_radius=int(ctx.params.get('closing_radius', 1)),
            min_hole_area=64,
            min_obj_area=48,
        )
        meta.update({'runtime_ms': (perf_counter() - t0) * 1000.0})
        return ExperimentOutput(prior_map=prior, mask=mask, meta=meta, status_level='success', prob_maps=prob_maps)


class UnetSmallPatchExperiment:
    info = ExperimentInfo(
        experiment_id='unet_small_patch',
        group='E',
        display_name='U-Net pequena por parches',
        description='Inferencia con checkpoint local; fallback a clasico si no hay checkpoint.',
        default_params={'checkpoint_path': '', 'threshold': 0.5},
        implementation_status='fallback',
        requirements_hint='torch + checkpoint_path valido',
    )

    def run(self, ctx: RunContext) -> ExperimentOutput:
        t0 = perf_counter()
        prior, meta = unet_small_patch_prior(ctx.image_rgb, ctx.labels, params=ctx.params)
        mask = threshold_and_clean(prior, thr=float(ctx.params.get('threshold', 0.5)), closing_radius=1, min_hole_area=40)
        meta.update({'runtime_ms': (perf_counter() - t0) * 1000.0})
        return ExperimentOutput(prior_map=prior, mask=mask, meta=meta, status_level=_status_from_meta(meta))


class AssistModelPredictExperiment:
    """Placeholder experiment for predictions made via the 'Predecir mascara' button.
    The actual prediction is handled by assist_models.predict_mask(); this entry
    only ensures the runs appear in 'Revisión de resultados'."""
    info = ExperimentInfo(
        experiment_id='assist_model_predict',
        group='F',
        display_name='Prediccion de mascara (modelo asistente)',
        description='Mascara generada por el modelo asistente entrenado.',
        default_params={},
        implementation_status='native',
    )

    def run(self, ctx: RunContext) -> ExperimentOutput:
        raise NotImplementedError('assist_model_predict se ejecuta via /api/assist-models/predict-mask')


def build_default_experiments() -> list:
    return [
        ExtraTreesPixelExperiment(),
        RandomForestPixelExperiment(),
        XGBoostPixelExperiment(),
        CatBoostPixelExperiment(),
        ExtraTreesBalancedExperiment(),
        ContextFeaturesExperiment(),
        ClassifierMorphMinExperiment(),
        UnetSmallPatchExperiment(),
        AssistModelPredictExperiment(),
    ]
