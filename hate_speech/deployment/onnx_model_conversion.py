# code to convert saved pytorch models to optimized+quantized onnx models for faster inference

import os
import shutil
from transformers.convert_graph_to_onnx import convert, optimize, quantize, verify
import argparse
import logging
from pathlib import Path

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_model_path_to_model_name(model_path):
    if 'bertweet' in model_path:
        return 'vinai/bertweet-base'
    elif 'roberta-base' in model_path:
        return 'roberta-base'
    elif 'DeepPavlov' in model_path:
        return 'DeepPavlov/bert-base-cased-conversational'


parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str)

args = parser.parse_args()

model_path = os.path.join(f'/scratch/mt4493/nigeria/trained_models/{args.model_name}',
                          'models', 'best_model')
onnx_path = os.path.join(model_path, 'onnx')

try:
    shutil.rmtree(onnx_path)  # deleting onxx folder and contents, if exists, conversion excepts
    logger.info('Remove existing ONNX folder and recreate empty')
    os.makedirs(onnx_path)
except:
    logger.info('no existing folder, creating one')
    os.makedirs(onnx_path)

logger.info('>> converting..')
convert(framework="pt",
        model=model_path,
        tokenizer=convert_model_path_to_model_name(model_path),
        output=Path(os.path.join(onnx_path, 'converted.onnx')),
        opset=11,
        pipeline_name='sentiment-analysis')

logger.info('>> ONNX optimization')
optimized_output = optimize(Path(os.path.join(onnx_path, 'converted.onnx')))
logger.info('>> Quantization')
quantized_output = quantize(optimized_output)

logger.info('>> Verification')
verify(Path(os.path.join(onnx_path, 'converted.onnx')))
verify(optimized_output)
verify(quantized_output)
