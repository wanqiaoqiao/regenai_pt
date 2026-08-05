from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from .forward_model import ForwardTransitionModel
from .inverse_recommender import recommend_inverse_treatments
from .model_registry import ModelRegistry
from .models.forward_interface import (
    CurrentStateProfile,
    ForwardTransitionModelInterface,
    TargetStateProfile,
)
from .registry import DatasetRegistry, ExperimentRegistry
from .training import (
    train_and_register_baseline,
    train_and_register_forward_transition,
    train_and_register_regenai_pt,
    train_and_register_regenai_pt_forward_transition,
)
from .treatments import load_treatment_library


def _parse_model_backend(value: str) -> str:
    if value == 'full_cpa_like':
        return 'regenai_pt'
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='ipsc-twin', description='iPSC production CLI')
    sub = parser.add_subparsers(dest='command', required=True)

    reg = sub.add_parser('registry', help='Registry operations')
    reg_sub = reg.add_subparsers(dest='registry_cmd', required=True)

    create_exp = reg_sub.add_parser('create-experiment', help='Create experiment record')
    create_exp.add_argument('--registry-dir', required=True, help='Directory for registry JSON files')
    create_exp.add_argument('--name', required=True)
    create_exp.add_argument('--objective', required=True)
    create_exp.add_argument('--owner', required=True)
    create_exp.add_argument('--status', default='active')
    create_exp.add_argument('--experiment-id', default=None)

    reg_ds = reg_sub.add_parser('register-dataset', help='Register AnnData dataset')
    reg_ds.add_argument('--registry-dir', required=True, help='Directory for registry JSON files')
    reg_ds.add_argument('--experiment-id', required=True)
    reg_ds.add_argument('--anndata-path', required=True)
    reg_ds.add_argument('--preprocessing-version', required=True)
    reg_ds.add_argument('--schema-version', required=True)

    list_exp = reg_sub.add_parser('list-experiments', help='List experiments')
    list_exp.add_argument('--registry-dir', required=True)

    train = sub.add_parser('train', help='Train a model')
    train.add_argument('--registry-dir', required=True)
    train.add_argument('--dataset-id', required=True)
    train.add_argument(
        '--model-type',
        required=True,
        type=_parse_model_backend,
        choices=['baseline', 'forward_transition', 'regenai_pt'],
    )
    train.add_argument(
        '--transition-model',
        type=_parse_model_backend,
        choices=['baseline', 'regenai_pt'],
        default='baseline',
    )
    train.add_argument('--output-dir', required=True)
    train.add_argument('--random-seed', type=int, default=0)
    train.add_argument('--max-epochs', type=int, default=5)
    train.add_argument('--batch-size', type=int, default=32)
    train.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    train.add_argument('--input-layer', choices=['X', 'raw_counts', 'log_normalized'], default='raw_counts')
    train.add_argument('--treatment-key')
    train.add_argument('--dose-key')
    train.add_argument('--covariate-keys', default='')
    train.add_argument('--n-latent', type=int, default=32)
    train.add_argument('--n-hidden', type=int, default=128)
    train.add_argument('--n-layers', type=int, default=2)
    train.add_argument('--dropout', type=float, default=0.1)
    train.add_argument('--learning-rate', type=float, default=1e-3)
    train.add_argument('--weight-decay', type=float, default=1e-6)
    train.add_argument('--gradient-clip-norm', type=float, default=5.0)
    train.add_argument('--reconstruction-loss', choices=['mse', 'nb', 'zinb'], default='mse')
    train.add_argument('--warmup-epochs', type=int, default=20)
    train.add_argument('--ramp-epochs', type=int, default=20)
    train.add_argument('--max-adversarial-weight', type=float, default=0.05)
    train.add_argument('--treatment-mode', choices=['single_round', 'combined_rounds'], default='single_round')
    train.add_argument('--control-treatment', default='control')

    predict_transition = sub.add_parser('predict-transition', help='Predict one-step RegenAI-PT transition')
    predict_transition.add_argument('--model-path', required=True)
    predict_transition.add_argument('--adata', required=True)
    predict_transition.add_argument('--treatment', required=True)
    predict_transition.add_argument('--dose', type=float)
    predict_transition.add_argument('--output-dir', required=True)

    simulate_sequence = sub.add_parser('simulate-sequence', help='Simulate two-step RegenAI-PT sequence')
    simulate_sequence.add_argument('--model-path', required=True)
    simulate_sequence.add_argument('--adata', required=True)
    simulate_sequence.add_argument('--round1-treatment', required=True)
    simulate_sequence.add_argument('--round1-dose', type=float)
    simulate_sequence.add_argument('--round2-treatment', required=True)
    simulate_sequence.add_argument('--round2-dose', type=float)
    simulate_sequence.add_argument('--output-dir', required=True)

    recommend = sub.add_parser('recommend', help='Inverse treatment recommendation')
    recommend.add_argument('--registry-dir', required=True)
    recommend.add_argument('--model-id', required=True)
    recommend.add_argument('--treatment-library', required=True)
    recommend.add_argument('--current-state-json')
    recommend.add_argument('--target-state-json')
    recommend.add_argument('--current-adata')
    recommend.add_argument('--target-adata')
    recommend.add_argument('--top-n', type=int, default=10)

    models = sub.add_parser('models', help='Model registry operations')
    models_sub = models.add_subparsers(dest='models_cmd', required=True)

    models_list = models_sub.add_parser('list', help='List models')
    models_list.add_argument('--registry-dir', required=True)

    models_desc = models_sub.add_parser('describe', help='Describe model')
    models_desc.add_argument('--registry-dir', required=True)
    models_desc.add_argument('model_id')

    return parser


def _dataset_record_by_id(dataset_registry: DatasetRegistry, dataset_id: str) -> dict:
    for rec in dataset_registry.list_datasets():
        if rec['dataset_id'] == dataset_id:
            return rec
    raise KeyError(f'Dataset not found: {dataset_id}')


def _parse_covariate_keys(raw: str) -> tuple[str, ...]:
    if not raw:
        return tuple()
    return tuple(part.strip() for part in raw.split(',') if part.strip())


def _load_regenai_pt_predictor(model_path: str | Path) -> tuple[Any, str]:
    from .models.regenai_pt_adapter import RegenAIPTForwardAdapter
    from .models.regenai_pt_trainer import RegenAIPTTrainer

    try:
        adapter = RegenAIPTForwardAdapter.load(model_path)
        if adapter.round1_trainer is not None:
            return adapter, 'adapter'
    except Exception:
        pass

    trainer = RegenAIPTTrainer.load(model_path)
    return trainer, 'trainer'


def _nearest_neighbor_summary(adata: ad.AnnData, predicted_expression: np.ndarray) -> dict[str, Any] | None:
    if adata.n_obs == 0:
        return None
    matrix = adata.X
    if hasattr(matrix, 'toarray'):
        matrix = matrix.toarray()
    observed = np.asarray(matrix, dtype=np.float32)
    pred_mean = np.asarray(predicted_expression, dtype=np.float32).mean(axis=0)
    distances = np.linalg.norm(observed - pred_mean[None, :], axis=1)
    nearest_idx = int(np.argmin(distances))
    summary: dict[str, Any] = {
        'nearest_neighbor_index': nearest_idx,
        'nearest_neighbor_distance': float(distances[nearest_idx]),
    }
    for column in ['stage_label', 'target_marker_score', 'stress_score', 'off_target_score', 'pseudotime', 'fate_probability']:
        if column in adata.obs.columns:
            summary[f'nearest_{column}'] = adata.obs.iloc[nearest_idx][column].item() if hasattr(adata.obs.iloc[nearest_idx][column], 'item') else adata.obs.iloc[nearest_idx][column]
    return summary


def _latent_to_csv(latent: np.ndarray, output_path: Path) -> None:
    frame = pd.DataFrame(latent, columns=[f'latent_{idx}' for idx in range(latent.shape[1])])
    frame.to_csv(output_path, index=False)


def _predicted_h5ad(predicted_expression: np.ndarray, template_adata: ad.AnnData, output_path: Path) -> None:
    ad.AnnData(
        X=np.asarray(predicted_expression, dtype=np.float32),
        obs=template_adata.obs.copy(),
        var=pd.DataFrame(index=template_adata.var_names.astype(str)),
    ).write_h5ad(output_path)


def _write_prediction_outputs(
    *,
    output_dir: Path,
    adata: ad.AnnData,
    predicted_expression: np.ndarray,
    predicted_latent: np.ndarray,
    summary: dict[str, Any],
    report_title: str,
    report_lines: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / 'predicted_expression.npy', np.asarray(predicted_expression, dtype=np.float32))
    _predicted_h5ad(np.asarray(predicted_expression, dtype=np.float32), adata, output_dir / 'predicted_expression.h5ad')
    _latent_to_csv(np.asarray(predicted_latent, dtype=np.float32), output_dir / 'predicted_latent.csv')
    (output_dir / 'prediction_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True))
    md = "\n".join([f'# {report_title}', '', *report_lines])
    (output_dir / 'simulation_report.md').write_text(md)


def _predict_transition_command(args: argparse.Namespace) -> int:
    predictor, predictor_type = _load_regenai_pt_predictor(args.model_path)
    adata = ad.read_h5ad(args.adata)
    output_dir = Path(args.output_dir)
    try:
        if predictor_type == 'adapter':
            prediction = predictor.predict_expression(
                adata,
                {'round1_treatment': args.treatment, 'round1_dose': args.dose, 'dose': args.dose, 'round_number': 1},
            )
            predicted_state = predictor.predict(
                adata,
                {'round1_treatment': args.treatment, 'round1_dose': args.dose, 'dose': args.dose, 'round_number': 1},
            )
            summary = {
                'mode': 'predict-transition',
                'predictor_type': predictor_type,
                'treatment': args.treatment,
                'dose': args.dose,
                'n_cells': int(adata.n_obs),
                'n_genes': int(adata.n_vars),
                'predicted_stage': predicted_state.stage,
                'uncertainty': float(predicted_state.uncertainty),
                'state_features': predicted_state.state_features,
                'mean_dose_scale': float(np.asarray(prediction['dose_scale']).mean()),
            }
        else:
            prediction = predictor.predict_adata(adata, treatment=args.treatment, dose=args.dose)
            summary = {
                'mode': 'predict-transition',
                'predictor_type': predictor_type,
                'treatment': args.treatment,
                'dose': args.dose,
                'n_cells': int(adata.n_obs),
                'n_genes': int(adata.n_vars),
                'mean_dose_scale': float(np.asarray(prediction['dose_scale']).mean()),
            }
    except KeyError as exc:
        raise ValueError(f"Unknown treatment '{args.treatment}' for loaded model artifact") from exc

    nn_summary = _nearest_neighbor_summary(adata, np.asarray(prediction['x_hat']))
    if nn_summary is not None:
        summary['nearest_neighbor_biological_score_summary'] = nn_summary

    _write_prediction_outputs(
        output_dir=output_dir,
        adata=adata,
        predicted_expression=np.asarray(prediction['x_hat']),
        predicted_latent=np.asarray(prediction['z_total']),
        summary=summary,
        report_title='Transition Prediction Report',
        report_lines=[
            f'- Treatment: {args.treatment}',
            f'- Dose: {args.dose}',
            f'- Cells: {adata.n_obs}',
            f'- Genes: {adata.n_vars}',
            f'- Predictor Type: {predictor_type}',
        ],
    )
    return 0


def _simulate_sequence_command(args: argparse.Namespace) -> int:
    predictor, predictor_type = _load_regenai_pt_predictor(args.model_path)
    if predictor_type != 'adapter' or not hasattr(predictor, 'simulate_sequence'):
        raise ValueError(
            'simulate-sequence requires a combined RegenAI-PT adapter artifact '
            'with both round1 and round2 trainers'
        )
    adata = ad.read_h5ad(args.adata)
    output_dir = Path(args.output_dir)
    try:
        result = predictor.simulate_sequence(
            adata,
            {
                'round1_treatment': args.round1_treatment,
                'round1_dose': args.round1_dose,
                'round2_treatment': args.round2_treatment,
                'round2_dose': args.round2_dose,
            },
        )
    except KeyError as exc:
        bad_treatment = args.round1_treatment if str(exc).strip("'") == args.round1_treatment else args.round2_treatment
        raise ValueError(f"Unknown treatment '{bad_treatment}' for loaded model artifact") from exc
    predicted_state = result.get('predicted_state')
    summary = {
        'mode': 'simulate-sequence',
        'predictor_type': predictor_type,
        'round1_treatment': args.round1_treatment,
        'round1_dose': args.round1_dose,
        'round2_treatment': args.round2_treatment,
        'round2_dose': args.round2_dose,
        'n_cells': int(adata.n_obs),
        'n_genes': int(adata.n_vars),
    }
    if predicted_state is not None:
        summary.update(
            {
                'predicted_stage': predicted_state.stage,
                'uncertainty': float(predicted_state.uncertainty),
                'state_features': predicted_state.state_features,
            }
        )
    nn_summary = _nearest_neighbor_summary(adata, np.asarray(result['x_hat']))
    if nn_summary is not None:
        summary['nearest_neighbor_biological_score_summary'] = nn_summary

    _write_prediction_outputs(
        output_dir=output_dir,
        adata=adata,
        predicted_expression=np.asarray(result['x_hat']),
        predicted_latent=np.asarray(result['round2']['z_total']),
        summary=summary,
        report_title='Sequence Simulation Report',
        report_lines=[
            f'- Round 1 Treatment: {args.round1_treatment}',
            f'- Round 1 Dose: {args.round1_dose}',
            f'- Round 2 Treatment: {args.round2_treatment}',
            f'- Round 2 Dose: {args.round2_dose}',
            f'- Cells: {adata.n_obs}',
            f'- Genes: {adata.n_vars}',
        ],
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == 'registry':
            registry_dir = Path(args.registry_dir)
            experiments_json = registry_dir / 'experiments.json'
            datasets_json = registry_dir / 'datasets.json'

            if args.registry_cmd == 'create-experiment':
                exp_reg = ExperimentRegistry(experiments_json)
                rec = exp_reg.create_experiment(
                    {
                        'experiment_id': args.experiment_id,
                        'name': args.name,
                        'objective': args.objective,
                        'owner': args.owner,
                        'status': args.status,
                    }
                )
                print(json.dumps(rec))
                return 0

            if args.registry_cmd == 'register-dataset':
                ds_reg = DatasetRegistry(datasets_json)
                rec = ds_reg.register_dataset(
                    anndata_path=args.anndata_path,
                    experiment_id=args.experiment_id,
                    preprocessing_version=args.preprocessing_version,
                    schema_version=args.schema_version,
                )
                print(json.dumps(rec))
                return 0

            if args.registry_cmd == 'list-experiments':
                exp_reg = ExperimentRegistry(experiments_json)
                print(json.dumps(exp_reg.list_experiments()))
                return 0

        if args.command == 'train':
            registry_dir = Path(args.registry_dir)
            ds_reg = DatasetRegistry(registry_dir / 'datasets.json')
            model_reg = ModelRegistry(registry_dir / 'models.json')
            ds = _dataset_record_by_id(ds_reg, args.dataset_id)

            if args.model_type == 'baseline':
                result = train_and_register_baseline(
                    dataset_path=ds['anndata_path'],
                    dataset_id=args.dataset_id,
                    dataset_version=ds['dataset_version'],
                    output_dir=args.output_dir,
                    model_registry=model_reg,
                    random_seed=args.random_seed,
                )
                print(json.dumps(result['model_record']))
                return 0

            if args.model_type == 'forward_transition':
                if args.transition_model == 'regenai_pt':
                    result = train_and_register_regenai_pt_forward_transition(
                        dataset_path=ds['anndata_path'],
                        dataset_id=args.dataset_id,
                        dataset_version=ds['dataset_version'],
                        output_dir=args.output_dir,
                        model_registry=model_reg,
                        random_seed=args.random_seed,
                        max_epochs=args.max_epochs,
                        batch_size=args.batch_size,
                        device=args.device,
                        input_layer=args.input_layer,
                        warmup_epochs=args.warmup_epochs,
                        ramp_epochs=args.ramp_epochs,
                        max_adversarial_weight=args.max_adversarial_weight,
                        gradient_clip_norm=args.gradient_clip_norm,
                        dataset_metadata=ds,
                    )
                else:
                    result = train_and_register_forward_transition(
                        dataset_path=ds['anndata_path'],
                        dataset_id=args.dataset_id,
                        dataset_version=ds['dataset_version'],
                        output_dir=args.output_dir,
                        model_registry=model_reg,
                        random_seed=args.random_seed,
                    )
                print(json.dumps(result['model_record']))
                return 0

            if args.model_type == 'regenai_pt':
                treatment_key = args.treatment_key or 'round1_treatment'
                dose_key = args.dose_key if args.dose_key is not None else ('round1_dose' if treatment_key == 'round1_treatment' else 'round2_dose')
                result = train_and_register_regenai_pt(
                    dataset_path=ds['anndata_path'],
                    dataset_id=args.dataset_id,
                    dataset_version=ds['dataset_version'],
                    output_dir=args.output_dir,
                    model_registry=model_reg,
                    config_kwargs={
                        'input_layer': args.input_layer,
                        'treatment_key': treatment_key,
                        'dose_key': dose_key,
                        'covariate_keys': _parse_covariate_keys(args.covariate_keys),
                        'n_latent': args.n_latent,
                        'n_hidden': args.n_hidden,
                        'n_layers': args.n_layers,
                        'dropout': args.dropout,
                        'max_epochs': args.max_epochs,
                        'batch_size': args.batch_size,
                        'learning_rate': args.learning_rate,
                        'weight_decay': args.weight_decay,
                        'gradient_clip_norm': args.gradient_clip_norm,
                        'reconstruction_loss': args.reconstruction_loss,
                        'warmup_epochs': args.warmup_epochs,
                        'ramp_epochs': args.ramp_epochs,
                        'max_adversarial_weight': args.max_adversarial_weight,
                        'random_seed': args.random_seed,
                        'device': args.device,
                        'control_treatment': args.control_treatment,
                    },
                    treatment_mode=args.treatment_mode,
                    random_seed=args.random_seed,
                    dataset_metadata=ds,
                )
                print(json.dumps(result['model_record']))
                return 0

        if args.command == 'predict-transition':
            return _predict_transition_command(args)

        if args.command == 'simulate-sequence':
            return _simulate_sequence_command(args)

        if args.command == 'recommend':
            model_reg = ModelRegistry(Path(args.registry_dir) / 'models.json')
            model_record = model_reg.describe_model(args.model_id)
            if model_record['model_type'] not in {'forward_transition', 'regenai_pt'}:
                raise ValueError(
                    'Inverse recommendation requires a forward_transition or RegenAI-PT model artifact'
                )

            training_config = model_record.get('training_config', {})
            transition_model = training_config.get('transition_model', 'baseline')
            model: ForwardTransitionModelInterface
            if model_record['model_type'] == 'regenai_pt' or transition_model == 'regenai_pt':
                from .models.regenai_pt_adapter import RegenAIPTForwardAdapter

                model = RegenAIPTForwardAdapter.load(model_record['artifact_path'])
            else:
                model = ForwardTransitionModel.load(model_record['artifact_path'])

            library = load_treatment_library(args.treatment_library)
            if args.current_adata:
                current_state = ad.read_h5ad(args.current_adata)
            elif args.current_state_json:
                current_state = CurrentStateProfile(state_features=json.loads(args.current_state_json))
            else:
                raise ValueError('Provide --current-state-json or --current-adata')

            if args.target_adata:
                target_state = ad.read_h5ad(args.target_adata)
            elif args.target_state_json:
                target_state = TargetStateProfile(state_features=json.loads(args.target_state_json))
            else:
                raise ValueError('Provide --target-state-json or --target-adata')

            recommendations = recommend_inverse_treatments(
                current_state=current_state,
                target_state=target_state,
                treatment_library=library,
                forward_model=model,
                top_n=args.top_n,
            )
            print(recommendations.to_json(orient='records'))
            return 0

        if args.command == 'models':
            model_reg = ModelRegistry(Path(args.registry_dir) / 'models.json')
            if args.models_cmd == 'list':
                print(json.dumps(model_reg.list_models()))
                return 0
            if args.models_cmd == 'describe':
                print(json.dumps(model_reg.describe_model(args.model_id)))
                return 0

        parser.error('Unsupported command')
        return 2
    except Exception as exc:
        logging.error('Command failed: %s', exc)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
