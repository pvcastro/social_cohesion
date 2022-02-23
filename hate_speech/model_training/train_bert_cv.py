from simpletransformers.classification import ClassificationModel
import argparse
import logging
import pandas as pd
import os
import ast
import json
import numpy as np
from pathlib import Path
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
from scipy.special import softmax
from scipy import interp
from typing import List
from pandas.api.types import is_string_dtype

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='bert')
    parser.add_argument("--model_name", type=str, default='DeepPavlov/bert-base-cased-conversational')
    parser.add_argument("--timestamp", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--learning_rate", type=float, default=4e-5)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--version", type=int, default=5)
    args = parser.parse_args()
    return args


def keep_rename_columns(df, columns_list):
    df = df[columns_list]
    df.columns = ['text', 'labels']
    df = df[df['labels'].notna()].reset_index(drop=True)
    return df


def get_env_var(varname, default):
    if os.environ.get(varname) != None:
        var = int(os.environ.get(varname))
        print(varname, ':', var)
    else:
        var = default
        print(varname, ':', var, '(Default)')
    return var


def prepare_filepath_for_storing_model(output_dir: str) -> str:
    """Prepare the filepath where the trained model will be stored.
    :param output_dir: Directory where to store outputs (trained models).
    :return: path_to_store_model: Path where to store the trained model.
    """
    path_to_store_model = os.path.join(output_dir, 'models')
    if not os.path.exists(path_to_store_model):
        os.makedirs(path_to_store_model)
    return path_to_store_model


def prepare_filepath_for_storing_best_model(path_to_store_model: str) -> str:
    path_to_store_best_model = os.path.join(path_to_store_model, 'best_model')
    if not os.path.exists(path_to_store_best_model):
        os.makedirs(path_to_store_best_model)
    return path_to_store_best_model


def read_json(filename: str):
    with open(filename) as f_in:
        return json.load(f_in)


def class_report(y_true, y_pred, y_score=None, average='macro'):
    if y_true.shape != y_pred.shape:
        print("Error! y_true %s is not the same shape as y_pred %s" % (
            y_true.shape,
            y_pred.shape)
              )
        return

    lb = LabelBinarizer()

    if len(y_true.shape) == 1:
        lb.fit(y_true)

    # Value counts of predictions
    labels, cnt = np.unique(
        y_true,
        return_counts=True)
    n_classes = len(labels)
    pred_cnt = pd.Series(cnt, index=labels)

    metrics_summary = metrics.precision_recall_fscore_support(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels)

    avg = list(metrics.precision_recall_fscore_support(
        y_true=y_true,
        y_pred=y_pred,
        average='weighted'))

    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index,
        columns=labels)

    support = class_report_df.loc['support']
    total = support.sum()
    class_report_df['avg / total'] = avg[:-1] + [total]

    class_report_df = class_report_df.T
    class_report_df['pred'] = pred_cnt
    class_report_df['pred'].iloc[-1] = total

    if not (y_score is None):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for label_it, label in enumerate(labels):
            fpr[label], tpr[label], _ = metrics.roc_curve(
                (y_true == label).astype(int),
                y_score[:, label_it])

            roc_auc[label] = metrics.auc(fpr[label], tpr[label])

        if average == 'micro':
            if n_classes <= 2:
                fpr["avg / total"], tpr["avg / total"], _ = metrics.roc_curve(
                    lb.transform(y_true).ravel(),
                    y_score[:, 1].ravel())
            else:
                fpr["avg / total"], tpr["avg / total"], _ = metrics.roc_curve(
                    lb.transform(y_true).ravel(),
                    y_score.ravel())

            roc_auc["avg / total"] = metrics.auc(
                fpr["avg / total"],
                tpr["avg / total"])

        elif average == 'macro':
            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([
                fpr[i] for i in labels]
            ))

            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in labels:
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])

            # Finally average it and compute AUC
            mean_tpr /= n_classes

            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr

            roc_auc["avg / total"] = metrics.auc(fpr["macro"], tpr["macro"])

        class_report_df['AUC'] = pd.Series(roc_auc)

    return class_report_df


if __name__ == '__main__':
    args = get_args_from_command_line()
    logger.info(f'Args: {args}')
    # load data
    folds = args.folds
    version = args.version
    folds_path = Path(args.data_path)
    assert folds_path.is_dir()
    for fold in range(folds):
        train_df = pd.read_csv(folds_path / f'train_v{version}_fold-{fold}.csv')
        eval_df = pd.read_csv(folds_path / f'test_v{version}_fold-{fold}.csv')
        num_labels = len(train_df['labels'].unique())
        # define paths and args
        output_dir = f"{args.output_dir}/{args.model_name.replace('/', '-')}/fold-{fold}"
        path_to_store_model = prepare_filepath_for_storing_model(output_dir=output_dir)
        path_to_store_best_model = prepare_filepath_for_storing_best_model(path_to_store_model)
        SLURM_JOB_ID = get_env_var('SLURM_JOB_ID', 1)
        classification_args = {
            'train_batch_size': args.batch_size,
            'gradient_accumulation_steps': 4,
            'overwrite_output_dir': True,
            'evaluate_during_training': True,
            'save_model_every_epoch': False,
            'save_eval_checkpoints': False,
            'output_dir': path_to_store_model,
            'best_model_dir': path_to_store_best_model,
            'evaluate_during_training_verbose': True,
            'num_train_epochs': args.num_train_epochs,
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            # "max_seq_length": 512,
            # "warmup_ratio": 0.2,
            # "scheduler": "constant_schedule",
            "do_lower_case": False,
            "use_early_stopping": True,
            "early_stopping_delta": 0,
            "early_stopping_metric": "eval_loss",
            "early_stopping_metric_minimize": True,
            "early_stopping_patience": 5,
            "tensorboard_dir": f"runs/{SLURM_JOB_ID}/",
            "manual_seed": args.seed}
        model = ClassificationModel(args.model_type, args.model_name, num_labels=num_labels, use_cuda=True,
                                    args=classification_args)
        # Train the model
        model.train_model(train_df, eval_df=eval_df, output_dir=path_to_store_model)
        logging.info("The training of the model is done")
        # Load best model (in terms of evaluation loss)
        train_args = read_json(filename=os.path.join(path_to_store_best_model, 'model_args.json'))
        best_model = ClassificationModel(args.model_type, path_to_store_best_model, args=train_args)
        logging.info("Loaded best model")
        # EVALUATION ON EVALUATION SET
        result, model_outputs, wrong_predictions = best_model.eval_model(eval_df)
        logging.info('Get output from best model')
        scores = np.array([softmax(element) for element in model_outputs])
        pred = np.argmax(scores, axis=1)
        result_report_macro_df = class_report(y_true=np.concatenate(eval_df[['labels']].to_numpy(), axis=0),
                                              y_pred=pred, y_score=scores, average='macro')
        logging.info('Built evaluation report')
        # Centralize evaluation results in a dictionary
        slurm_job_timestamp = args.timestamp
        slurm_job_id = SLURM_JOB_ID

        # Save evaluation results on eval set
        # if "/" in args.model_name:
        #     args.model_name = args.model_name.replace('/', '-')
        path_to_store_eval_results = os.path.join(f'{output_dir}_{str(slurm_job_id)}',
                                                  f'evaluation.csv')
        if not os.path.exists(os.path.dirname(path_to_store_eval_results)):
            os.makedirs(os.path.dirname(path_to_store_eval_results))
        result_report_macro_df.to_csv(path_to_store_eval_results)
        logging.info(
            "The evaluation on the evaluation set is done. The results were saved at {}".format(path_to_store_eval_results))
        # Save scores
        eval_df['score'] = scores.tolist()
        path_to_store_eval_scores = os.path.join(f'{output_dir}_{str(slurm_job_id)}',
                                                 f'scores.csv')
        if not os.path.exists(os.path.dirname(path_to_store_eval_scores)):
            os.makedirs(os.path.dirname(path_to_store_eval_scores))
        eval_df.to_csv(path_to_store_eval_scores, index=False)
        logging.info("The scores for the evaluation set were saved at {}".format(path_to_store_eval_scores))
