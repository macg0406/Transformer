# import tensorflow as tf

import pickle
import os
import json
import time
import joblib
import random
import array

from tqdm import tqdm
import numpy as np
# from transformer_tf import Transformer, CustomSchedule, create_masks
# from vocabulary import tokenization
from Tokenize import SPMTokenize
from Optim import CosineWithRestarts
import torch
from Models import Transformer
from Batch import create_masks
# import torch.nn.functional as F

EPOCHS = 30
MAX_TOTAL_LENGTH = 3500
d_model = 256
d_ff = 2048
num_layers = 3
num_heads = 8
dropout_rate = 0.1
# checkpoint_path = "./checkpoints/train_en2zh"
data_dump_path = "datasets_en2zh.pkl"
checkpoint_path = "weights/model_weights"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
#     from_logits=True, reduction='none')


def LOG(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S ||| ", time.localtime()), message)


# def loss_function(real, pred):
#     mask = tf.math.logical_not(tf.math.equal(real, 0))
#     loss_ = loss_object(real, pred)

#     mask = tf.cast(mask, dtype=loss_.dtype)
#     loss_ *= mask

#     return tf.reduce_mean(loss_)


def load_file(filename, result_list, tokenizer_en, tokenizer_zh):
    with open(filename) as corpus:
        for line in corpus:
            data = json.loads(line)
            ids_en = tokenizer_en.convert_sent_to_ids(data["english"])
            ids_zh = tokenizer_zh.convert_sent_to_ids(data["chinese"])
            ids_en = array.array("H", ids_en)
            ids_zh = array.array("H", ids_zh)
            result_list.append((ids_en, ids_zh))
            if len(result_list) % 100000 == 0:
                print(len(result_list))
            

def encode_sentence(lang1, lang2, tokenizer_1, tokenizer_2):
    token_ids_1 = tokenizer_1.convert_sent_to_ids(lang1)
    token_ids_2 = tokenizer_2.convert_sent_to_ids(lang2)
    return token_ids_1, token_ids_2


def batchify(data_id_list, max_total_length):
    batch_datas = []
    data_id_list.sort(key=lambda x: (len(x[1]), len(x[0])))
    current_bs = 0
    max_len_en = 0
    max_len_zh = 0
    current_batch = []
    for data_ids in data_id_list:
        current_bs += 1
        len_en, len_zh = len(data_ids[0]), len(data_ids[1])
        temp_max_len_en = max(len_en, max_len_en)
        temp_max_len_zh = max(len_zh, max_len_zh)
        if (temp_max_len_en + temp_max_len_zh) * current_bs > max_total_length:
            batch_datas.append(current_batch)
            max_len_en = len_en
            max_len_zh = len_zh
            current_bs = 1
            current_batch = []
        else:
            max_len_en = temp_max_len_en
            max_len_zh = temp_max_len_zh
        current_batch.append(data_ids)
    if current_batch:
        batch_datas.append(current_batch)
    return batch_datas


def read_data():
    if os.path.exists(data_dump_path):
        LOG("Load data from %s." % data_dump_path)
        with open(data_dump_path, "rb") as fin:
            data = pickle.load(fin)
        return data
    
    tokenizer_zh = SPMTokenize(lang="en")  # tokenization.FullTokenizer("vocabulary/zh_vocab.txt")
    tokenizer_en = SPMTokenize(lang="zh")  # tokenization.FullTokenizer("vocabulary/en_vocab.txt")
    vocab_size_en = tokenizer_en.vocab_size
    vocab_size_zh = tokenizer_zh.vocab_size
    
    train_list = []
    valid_list = []
    load_file("data/translation2019zh_valid.json", valid_list, tokenizer_en, tokenizer_zh)
    LOG(len(valid_list))
    # load_file("data/translation2019zh_valid.json", train_list)
    load_file("data/translation2019zh_train.json", train_list, tokenizer_en, tokenizer_zh)
    LOG(len(train_list))
    
    input_vocab_size = vocab_size_en
    target_vocab_size = vocab_size_zh
    with open(data_dump_path, "wb") as fin:
        pickle.dump((train_list, valid_list, input_vocab_size,
                     target_vocab_size, tokenizer_en.pad_id, tokenizer_zh.pad_id), fin)
    # joblib.dump((train_list, valid_list, input_vocab_size,
    #              target_vocab_size, tokenizer_en.pad_id, tokenizer_zh.pad_id), data_dump_path)
    return train_list, valid_list, input_vocab_size, target_vocab_size, tokenizer_en.pad_id, tokenizer_zh.pad_id


def batch_to_tensor(batch_data, inp_pad_id, tar_pad_id):
    batch_size = len(batch_data)
    random.shuffle(batch_data)
    maxlen_0 = max(len(x[0]) for x in batch_data)
    maxlen_1 = max(len(x[1]) for x in batch_data)
    inp = np.ones((batch_size, maxlen_0), dtype=np.int32) * inp_pad_id
    tar = np.ones((batch_size, maxlen_1), dtype=np.int32) * tar_pad_id
    for i, (en, zh) in enumerate(batch_data):
        inp[i, :len(en)] = en
        tar[i, :len(zh)] = zh
    return torch.tensor(inp), torch.tensor(tar)


def get_tensor_batch(data_list, inp_pad_id, tar_pad_id):
    random.shuffle(data_list)
    for batch_data in data_list:
        yield batch_to_tensor(batch_data, inp_pad_id, tar_pad_id)


def load_embeddings(path):
    embed_data = joblib.load(path)
    vocab_size, d_model = embed_data.shape
    embedings = np.random.random((vocab_size+2, d_model))
    embedings[:vocab_size, :] = embed_data
    return embedings


def main():
    train_id_list, valid_id_list, input_vocab_size, target_vocab_size, inp_pad_id, tag_pad_id = read_data()
    LOG("Load data finished, %d training, %d validation" %
        (len(train_id_list), len(valid_id_list)))
    train_dataset = batchify(train_id_list, MAX_TOTAL_LENGTH)
    val_dataset = batchify(valid_id_list, MAX_TOTAL_LENGTH)
    nbatch_train, nbatch_val = len(train_dataset), len(val_dataset)
    LOG(" %d batches of training data, %d batches of validation data." %
        (nbatch_train, nbatch_val))
    # en_embed_data = load_embeddings("vocabulary/en_embedding.dat")
    # zh_embed_data = load_embeddings("vocabulary/zh_embedding.dat")

    # d_model = dff = zh_embed_data.shape[-1]
    # learning_rate = CustomSchedule(d_model)
    # optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
    # sched = CosineWithRestarts(optimizer, T_max=train_len)
    # optimizer = tf.keras.optimizers.Adam(learning_rate)
    
    # train_loss = tf.keras.metrics.Mean(name='train_loss')
    # train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    #     name='train_accuracy')
    # transformer = Transformer(num_layers, num_heads, dff,
    #                           # input_vocab_size, target_vocab_size,
    #                           en_embed_data, zh_embed_data,
    #                           pe_input=input_vocab_size,
    #                           pe_target=target_vocab_size,
    #                           rate=dropout_rate)
    model = Transformer(input_vocab_size, target_vocab_size, d_model, num_layers, num_heads, dropout_rate)
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        LOG(f"Load model from {checkpoint_path}.")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
    loss_func = torch.nn.functional.cross_entropy
    # ckpt = tf.train.Checkpoint(transformer=transformer,
    #                            optimizer=optimizer)

    # ckpt_manager = tf.train.CheckpointManager(
    #     ckpt, checkpoint_path, max_to_keep=5)

    # 如果检查点存在，则恢复最新的检查点。
    # if ckpt_manager.latest_checkpoint:
    #     ckpt.restore(ckpt_manager.latest_checkpoint)
    #     print('Latest checkpoint restored!!')

    # train_step_signature = [
    #     tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    #     tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    # ]
    # @tf.function(input_signature=train_step_signature)
    def train_step(src, tar, model, loss_func, optimizer):
        trg_input = tar[:, :-1]
        tar_real = tar[:, 1:].contiguous().view(-1).type(torch.LongTensor)

        src_mask, trg_mask = create_masks(src, trg_input, inp_pad_id, tag_pad_id, device=device)
        preds = model(src, trg_input, src_mask, trg_mask)
        optimizer.zero_grad()
        loss = loss_func(preds.view(-1, preds.size(-1)), tar_real, ignore_index=tag_pad_id)
        loss.backward()
        optimizer.step()
        return loss.item()
        # train_loss(loss)
        # train_accuracy(tar_real, predictions)
        # if opt.SGDR == True: 
        #     opt.sched.step()


    def evaluate_step(src, tar, model, loss_func):
        trg_input = tar[:, :-1]
        tar_real = tar[:, 1:].contiguous().view(-1).type(torch.LongTensor)

        src_mask, trg_mask = create_masks(src, trg_input, inp_pad_id, tag_pad_id, device=device)
        preds = model(src, trg_input, src_mask, trg_mask)
        optimizer.zero_grad()
        loss = loss_func(preds.view(-1, preds.size(-1)), tar_real, ignore_index=tag_pad_id)
        return loss.item()

    LOG("Params MAX_TOTAL_LENGTH:%d, d_model:%d, dff:%d, num_layers:%d, num_heads:%d." %
        (MAX_TOTAL_LENGTH, d_model, d_ff, num_layers, num_heads))
    for epoch in range(EPOCHS):
        start = time.time()

        loss_list = []
        model.train()
        for batch, (inp, tar) in enumerate(get_tensor_batch(train_dataset, inp_pad_id, tag_pad_id)):
            loss_value = train_step(inp, tar, model, loss_func, optimizer)
            loss_list.append(loss_value)

            if batch % 1000 == 0:
                LOG('Epoch {} Batch {}/{} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch, nbatch_train, sum(loss_list)/len(loss_list), 0.0))
            if batch % 10000 == 0:
                torch.save(model.state_dict(), checkpoint_path)
        torch.save(model.state_dict(), checkpoint_path)
        with torch.no_grad():
            for batch_data in val_dataset:
                inp, tar = batch_to_tensor(batch_data, inp_pad_id, tag_pad_id)
                evaluate_step(inp, tar, model, loss_func)

        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


if __name__ == "__main__":
    # main()
    read_data()
