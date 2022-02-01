import os
import torch
import onnxruntime as ort
import pandas as pd
import numpy as np
import os
import time
import torch.nn.functional as F
import onnx
import getpass
from transformers import AutoTokenizer
import time
import pyarrow.parquet as pq
from glob import glob
import os
import numpy as np
import argparse
import logging
import socket
import multiprocessing
from functools import reduce


parser = argparse.ArgumentParser()

parser.add_argument("--model_folder", type=str)
parser.add_argument("--input_path", type=str)
parser.add_argument("--output_path", type=str)
parser.add_argument("--drop_duplicates", type=bool, help="drop duplicated tweets from parquet files", default=False)
parser.add_argument("--resume", type=int, help="resuming a run, 0 or 1")
# parser.add_argument("--log_path", type=str, help="resuming a run, 0 or 1")



args = parser.parse_args()

logging.basicConfig(
                    # filename=f'{args.log_path}.log',
                    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
print('libs loaded')

print(args)


global_start = time.time()

####################################################################################################################################
# HELPER FUNCTIONS
####################################################################################################################################

# inference
def get_tokens(tokens_dict, i):
    i_tokens_dict = dict()
    for key in ['input_ids', 'token_type_ids', 'attention_mask']:
        i_tokens_dict[key] = tokens_dict[key][i]
    tokens = {name: np.atleast_2d(value) for name, value in i_tokens_dict.items()}
    return tokens


def inference(onnx_model, model_dir, examples):
    quantized_str = ''
    if 'quantized' in onnx_model:
        quantized_str = 'quantized'
    onnx_inference = []
    #     pytorch_inference = []
    # onnx session
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 1
    # options.inter_op_num_threads = multiprocessing.cpu_count()

    print(onnx_model)
    ort_session = ort.InferenceSession(onnx_model, options)

    # pytorch pretrained model and tokenizer
    if 'bertweet' in onnx_model:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, normalization=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

    tokenizer_str = "TokenizerFast"

    print("**************** {} ONNX inference with batch tokenization and with {} tokenizer****************".format(
        quantized_str, tokenizer_str))
    start_onnx_inference_batch = time.time()
    start_batch_tokenization = time.time()
    tokens_dict = tokenizer.batch_encode_plus(examples, max_length=128)
    total_batch_tokenization_time = time.time() - start_batch_tokenization
    total_inference_time = 0
    total_build_label_time = 0
    for i in range(len(examples)):
        """
        Onnx inference with batch tokenization
        """

        if i % 100 == 0:
            print(f'[inference: {i} out of {str(len(examples))}')

        tokens = get_tokens(tokens_dict, i)
        # inference
        start_inference = time.time()
        ort_outs = ort_session.run(None, tokens)
        total_inference_time = total_inference_time + (time.time() - start_inference)
        # build label
        start_build_label = time.time()
        torch_onnx_output = torch.tensor(ort_outs[0], dtype=torch.float32)
        print(torch_onnx_output)
        # if feature == 'sexism':
        onnx_logits = F.softmax(torch_onnx_output, dim=1)
        logits_label = torch.argmax(onnx_logits, dim=1)
        label = logits_label.detach().cpu().numpy()
        #         onnx_inference.append(label[0])
        onnx_inference.append(onnx_logits.detach().cpu().numpy()[0].tolist())
        total_build_label_time = total_build_label_time + (time.time() - start_build_label)
    #         print(i, label[0], onnx_logits.detach().cpu().numpy()[0].tolist(), type(onnx_logits.detach().cpu().numpy()[0]) )
    #     elif feature == 'empathy':
    #         onnx_inference.append(torch_onnx_output.item())
        # break #DEBUG

    end_onnx_inference_batch = time.time()
    print("Total batch tokenization time (in seconds): ", total_batch_tokenization_time)
    print("Total inference time (in seconds): ", total_inference_time)
    print("Total build label time (in seconds): ", total_build_label_time)
    print("Duration ONNX inference (in seconds) with {} and batch tokenization: ".format(tokenizer_str),
          end_onnx_inference_batch - start_onnx_inference_batch,
          (end_onnx_inference_batch - start_onnx_inference_batch) / len(examples))

    return onnx_inference


def get_env_var(varname, default):
    if os.environ.get(varname) != None:
        var = int(os.environ.get(varname))
        print(varname, ':', var)
    else:
        var = default
        print(varname, ':', var, '(Default)')
    return var


# Choose Number of Nodes To Distribute Credentials: e.g. jobarray=0-4, cpu_per_task=20, credentials = 90 (<100)
SLURM_ARRAY_TASK_ID = get_env_var('SLURM_ARRAY_TASK_ID', 0)
SLURM_ARRAY_TASK_COUNT = get_env_var('SLURM_ARRAY_TASK_COUNT', 1)
SLURM_JOB_ID = get_env_var('SLURM_JOB_ID', 1)
print('Hostname:', socket.gethostname())
print('SLURM_ARRAY_TASK_ID', SLURM_ARRAY_TASK_ID)
print('SLURM_ARRAY_TASK_COUNT', SLURM_ARRAY_TASK_COUNT)
# ####################################################################################################################################
# # loading data
# ####################################################################################################################################

path_to_data = args.input_path

print('Load random Tweets:')

start_time = time.time()

final_output_path = args.output_path

if not os.path.exists(os.path.join(final_output_path)):
    print('>>>> directory doesnt exists, creating it')
    os.makedirs(os.path.join(final_output_path))

input_files_list = glob(os.path.join(path_to_data, '*.parquet'))
print(input_files_list)
"""
creating a list of unique file ids assuming this file name structure:
/scratch/spf248/twitter/data/user_timeline/user_timeline_text_preprocessed/part-00000-52fdb0a4-e509-49fe-9f3a-d809594bba7d-c000.snappy.parquet
in this case:
unique_intput_file_id_list will contain 00000-52fdb0a4-e509-49fe-9f3a-d809594bba7d
filename_prefix is /scratch/spf248/twitter/data/user_timeline/user_timeline_text_preprocessed/part-
filename_suffix is -c000.snappy.parquet
"""
unique_intput_file_id_list = [filename.split('part-')[1].split('-c000')[0]
                              for filename in input_files_list]
filename_prefix = input_files_list[0].split('part-')[0]
filename_suffix = input_files_list[0].split('part-')[1].split('-c000')[1]

already_processed_output_files = glob(os.path.join(final_output_path, '*.parquet'))
already_processed_file_id_list = [filename.split('part-')[1].split('-c000')[0]
                              for filename in already_processed_output_files]
unique_already_processed_file_id_list = list(dict.fromkeys(already_processed_file_id_list))

if args.resume == 1:
    unique_ids_remaining = list(set(unique_intput_file_id_list) - set(unique_already_processed_file_id_list))
    unique_ids_remaining = list(dict.fromkeys(unique_ids_remaining))
    files_remaining = [filename_prefix+'part-'+filename+'-c000'+filename_suffix for filename in unique_ids_remaining]
    print(files_remaining[:3])
    print(len(files_remaining), len(unique_intput_file_id_list), len(unique_already_processed_file_id_list))
else:
    files_remaining = input_files_list
print('resume', args.resume, len(files_remaining), len(unique_intput_file_id_list),
      len(unique_already_processed_file_id_list))

paths_to_random = list(np.array_split(
        files_remaining,
        SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID]
    )
print('#files in paths_to_random', len(paths_to_random))


tweets_random = pd.DataFrame()



TOTAL_NUM_TWEETS = 0
for file in paths_to_random:
    print(file)
    filename_without_extension = os.path.splitext(os.path.splitext(file.split('/')[-1])[0])[0]
    print('filename_without_extension')


    tweets_random = pd.read_parquet(file)
    print(tweets_random.shape)

    # tweets_random = tweets_random.head(10) #DEBUG

    print('load random sample:', str(time.time() - start_time), 'seconds')
    print(tweets_random.shape)


    if args.drop_duplicates:
        print('dropping duplicates:')
        start_time = time.time()
        tweets_random = tweets_random.drop_duplicates('text')
        print('drop duplicates:', str(time.time() - start_time), 'seconds')
        print(tweets_random.shape)

    start_time = time.time()
    print('converting to list')
    examples = tweets_random.text.values.tolist()
    # examples = examples[0] #DEBUG
    TOTAL_NUM_TWEETS = TOTAL_NUM_TWEETS + len(examples)

    print('convert to list:', str(time.time() - start_time), 'seconds')
    all_predictions_random_df_list = []

    # print('\n\n!!!!!column', column)
    loop_start = time.time()
    model_path = os.path.join('/scratch/mt4493/nigeria/trained_models', args.model_folder, 'models', 'best_model')

    print(model_path)
    onnx_path = os.path.join(model_path, 'onnx')
    print(onnx_path)

    ####################################################################################################################################
    # TOKENIZATION and INFERENCE
    ####################################################################################################################################
    print('Predictions of random Tweets:')
    start_time = time.time()
    onnx_labels = inference(onnx_model=os.path.join(onnx_path, 'converted-optimized-quantized.onnx'),
                            model_dir=model_path,
                            examples=examples)

    print('time taken:', str(time.time() - start_time), 'seconds')
    print('per tweet:', (time.time() - start_time) / tweets_random.shape[0], 'seconds')

    ####################################################################################################################################
    # SAVING
    ####################################################################################################################################
    print('Save Predictions of random Tweets:')
    start_time = time.time()

    # create dataframe containing tweet id and probabilities
    predictions_random_df = pd.DataFrame(data=onnx_labels)#, columns=['first', 'second'])
    predictions_random_df = predictions_random_df.set_index(tweets_random.tweet_id)
    # reformat dataframe
    # predictions_random_df = predictions_random_df[['second']]
    # predictions_random_df.columns = ['score']
    # predictions_random_df.columns = [args.predicted_feature]
    # predictions_random_df = tweets_random.join(predictions_random_df)

    print(predictions_random_df.head())


    # all_predictions_random_df_list.append(predictions_random_df)
    #
    # # break  # DEBUG column
    #
    # all_columns_df = reduce(lambda x,y: pd.merge(x , y, left_on=['tweet_id'], right_on=['tweet_id'] ,how='inner'),
    #                         all_predictions_random_df_list
    #                         )

    # print('!!all_columns_df', all_columns_df.head())
    # print('!!shapes', all_columns_df.shape, [df.shape for df in all_predictions_random_df_list])

    predictions_random_df.columns = ['_', 'score']
    predictions_random_df = predictions_random_df[['score']]
    predictions_random_df.to_parquet(
        os.path.join(final_output_path,
                     filename_without_extension + str(getpass.getuser()) + '_random' + '-' + str(SLURM_JOB_ID)
                     + '.parquet'))

    print('saved to:',
          # column,
          SLURM_ARRAY_TASK_ID,
          SLURM_JOB_ID,
          SLURM_ARRAY_TASK_COUNT,
          filename_without_extension,
          os.path.join(final_output_path,
                                      filename_without_extension + str(getpass.getuser()) + '_random' + '-' + str(SLURM_JOB_ID) + '.parquet'),
          str(time.time() - start_time)
        )

    print('>>>>> completed', filename_without_extension)

    print('save time taken:', str(time.time() - start_time), 'seconds')

    print('file loop:', filename_without_extension, str(time.time() - loop_start), 'seconds', (time.time() -
                                                                                                  loop_start) / len(examples))
    # break #DEBUG parquet file

if TOTAL_NUM_TWEETS > 0:
    print('full loop:', str(time.time() - global_start), 'seconds',
          (time.time() - global_start) / TOTAL_NUM_TWEETS)

print('>>done')
