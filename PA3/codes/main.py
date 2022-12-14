from psutil import cpu_count
import tqdm
import torch
import torch.nn as nn
import json
import numpy as np
import time
import random
import argparse
import torch
from torch import optim
from tokenizer import get_tokenizer
import os
from model_tfmr import TfmrLMHeadModel, TransposeLinear

random.seed(1229)
torch.manual_seed(1229)
torch.cuda.manual_seed_all(1229)
np.random.seed(1229)
from configuration import ModelConfig
import wandb
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

#! 之后写 pipeline 的时候需要注意添加 test list
#! test 的名字和  train 的 ckpt 完全一致。但是 wandb_run_name 会带上 temperature, top_k, top_p 等参数
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name", type=str, default=None, help="Experiment name. Used for wandb."
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="config_base.json",
        help="Path to the configuration file. Default: ./config.json",
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default="./tokenizer",
        help="Tokenizer file directory. Default: ./tokenizer",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of training epoch. Default: 20",
    )
    parser.add_argument(
        "--cpu_count",
        type=int,
        default=20,
        help="Number of CPU cores for evaluation. Default: 20",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="The number of batch_size. Default: 32",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate during optimization. Default: 1e-4",
    )
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Evaluate the model with the specified name. Default: None",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Data directory. Default: ../data",
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default="./train_ckpt",
        help="Training directory for saving model. Default: ./train_ckpt",
    )
    #! 需要对比 pretrain 与 skcratch 的结果
    parser.add_argument(
        "--pretrain_dir",
        type=str,
        default=None,
        help="Pre-Training directory for loading pretrained model. Default: None",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=35,
        help="Maximum length for training/inference. Default: 35",
    )
    parser.add_argument(
        "--decode_strategy",
        type=str,
        choices=["random", "top-p", "top-k"],
        default="random",
        help='The strategy for decoding. Can be "random", "top-p" or "top-k". Default: random',
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="The temperature for decoding. Default: 1",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.8,
        help="The p for top-p sampling. Default: 0.8",
    )
    parser.add_argument(
        "--top_k", type=int, default=40, help="The k for top-k sampling. Default: 40"
    )
    parser.add_argument(
        "--using_wandb",
        action="store_true",
        help="Whether to use W&B logging. Default: False",
    )
    parser.add_argument(
        "--waiting_epoch",
        type=int,
        default=3,
        help="The epoch to start waiting for the tarining to end. Default: 5",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=3,
        help="The number of layers for the transformer. Default: 3",
    )
    parser.add_argument(
        "--extract_layer",
        type=int,
        default=0,
        help="The number of extract layers for the transformer. Default: 0",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=12,
        help="The number of heads for the transformer. Default: 12",
    )
    args = parser.parse_args()
    return (
        args,
        args.name,
        args.model_config,
        args.tokenizer_dir,
        args.num_epochs,
        args.cpu_count,
        args.batch_size,
        args.learning_rate,
        args.test,
        args.data_dir,
        args.train_dir,
        args.pretrain_dir,
        args.max_len,
        args.decode_strategy,
        args.temperature,
        args.top_p,
        args.top_k,
        args.using_wandb,
        args.waiting_epoch,
        args.num_layers,
        args.extract_layer,
        args.num_heads,
    )


def _sentence_bleu(ele):
    return sentence_bleu(
        ele[0], ele[1], weights=ele[2], smoothing_function=SmoothingFunction().method1
    )


def fast_evaluate(model, data, batch_size, PAD_ID, device):
    model.eval()
    st, ed, all_loss = 0, 0, []
    while ed < len(data):
        st, ed = ed, (ed + batch_size) if (ed + batch_size < len(data)) else len(data)
        with torch.no_grad():
            input_ids = torch.tensor(data[st:ed]).to(device)
            ce_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            # TODO START
            #! Warning
            tgt_ids = input_ids[:, 1:]
            input_ids = input_ids[:, :-1]
            lm_logits = model(input_ids)["logits"]

            loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.contiguous().view(-1))
            pad_pos = torch.eq(tgt_ids, PAD_ID).to(torch.float).to(device)
            pad_pos = torch.cat([torch.zeros([ed-st, 1]).to(device), pad_pos[:, :-1]], 1)
            loss_mask = 1. - pad_pos
            loss = torch.sum(loss.view(input_ids.size()[0], -1) * loss_mask, 1) / (torch.sum(loss_mask, 1) + 1e-20)

            # TODO END
            all_loss += loss.cpu().numpy().tolist()
    loss = np.mean(all_loss)
    ppl = np.exp(loss)
    model.train()
    return loss, ppl


def evaluate(gen_ids, truth_ids, cpu_count=20):
    from multiprocessing import Pool

    # reference: https://superfastpython.com/multiprocessing-pool-python/

    assert len(gen_ids) == len(truth_ids)
    sample_hyps_num = len(gen_ids)
    res = {}
    for ngrams in [4]:
        print("computing BLEU-%d" % ngrams)
        bleu_irl_fw, bleu_irl_bw = [], []
        weights = np.ones(ngrams) / ngrams

        tasks = ((truth_ids, gen_ids[i], weights) for i in range(sample_hyps_num))
        pool = Pool(cpu_count)
        values = pool.imap_unordered(_sentence_bleu, tasks, chunksize=20)
        values = tqdm.tqdm(values, total=sample_hyps_num)
        for ans in values:
            bleu_irl_fw.append(ans)
        pool.close()
        pool.join()

        tasks = ((gen_ids, truth_ids[i], weights) for i in range(sample_hyps_num))
        pool = Pool(cpu_count)
        values = pool.imap_unordered(_sentence_bleu, tasks, chunksize=20)
        values = tqdm.tqdm(values, total=sample_hyps_num)
        for ans in values:
            bleu_irl_bw.append(ans)
        pool.close()
        pool.join()

        fw_bleu = 1.0 * sum(bleu_irl_fw) / len(bleu_irl_fw)
        bw_bleu = 1.0 * sum(bleu_irl_bw) / len(bleu_irl_bw)
        if fw_bleu + bw_bleu > 0:
            fw_bw_bleu = 2.0 * bw_bleu * fw_bleu / (fw_bleu + bw_bleu)
        else:
            fw_bw_bleu = 0

        res.update(
            {
                "fw-bleu-%d" % ngrams: fw_bleu,
                "bw-bleu-%d" % ngrams: bw_bleu,
                "fw-bw-bleu-%d" % ngrams: fw_bw_bleu,
            }
        )
    return res


def load_data(path, tokenizer, PAD_ID, field_list=["train", "dev", "test"], max_len=40):
    data, data_remove_pad = {}, {}
    for name in field_list:
        data[name], data_remove_pad[name] = [], []
        with open("%s/%s.txt" % (path, name)) as fin:
            for line in fin:
                tokens = tokenizer.encode(line.strip())
                if len(tokens) < max_len:
                    data[name].append(
                        [PAD_ID] + tokens + [PAD_ID] * (max_len - len(tokens))
                    )
                else:
                    data[name].append([PAD_ID] + tokens[:max_len])
                data_remove_pad[name].append(tokens)
    return data, data_remove_pad


def get_init_weights_func(config):
    def init_weights(module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, TransposeLinear)):
            module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    return init_weights


if __name__ == "__main__":
    (
        args,
        name,
        model_config,
        tokenizer_dir,
        num_epochs,
        cpu_count,
        batch_size,
        learning_rate,
        test,
        data_dir,
        train_dir,
        pretrain_dir,
        max_len,
        decode_strategy,
        temperature,
        top_p,
        top_k,
        using_wandb,
        waiting_epoch,
        num_layers,
        extract_layer,
        num_heads,
    ) = parser_args()
    if extract_layer != 0:
        extraction_dict = {1: "first", 2: "last", 3: "skip"}
        wandb_run_name = f"extraction_{extraction_dict[extract_layer]}"
    elif args.test is None:
        if args.pretrain_dir is None:
            print("Created model with fresh parameters.")
            with open(args.model_config) as fin:
                model_config = json.load(fin)
                print("layer num is " + str(args.num_layers))
                model_config["n_layer"] = args.num_layers
                model_config["n_head"] = args.num_heads
                config = ModelConfig(**model_config)
                if args.num_heads != 12:
                     wandb_run_name = f"{model_config['n_layer']}_{args.batch_size}_{args.num_heads}"
                else:
                    wandb_run_name = f"{model_config['n_layer']}_{args.batch_size}"
        else:
            #! 用 full 的话，batch_size = 64 会炸
            try:
                if "\\" in pretrain_dir:
                    wandb_run_name = str(pretrain_dir).split("\\")[-1][:-4]
                elif "/" in pretrain_dir:
                    wandb_run_name = str(pretrain_dir).split("/")[-1][:-4]
            except:
                wandb_run_name = pretrain_dir

            wandb_run_name = wandb_run_name + f"_bs{args.batch_size}"
    else:
        try:
            if "\\" in test:
                test_model = str(test).split("\\")[-1][:-4]
            elif "/" in test:
                test_model = str(test).split("/")[-1][:-4]
        except:
            test_model = test
        wandb_run_name = f"{test_model}_{decode_strategy}_{temperature}_{top_p}_{top_k}"

    args.name = wandb_run_name
    print(f"wandb name is {wandb_run_name}")

    if using_wandb:
        wandb.init(project="Transformer-Gen", entity="eren-zhao", name=wandb_run_name)
        wandb.config = {
            "name": args.name,
            "model_config": model_config,
            "tokenizer_dir": tokenizer_dir,
            "num_epochs": num_epochs,
            "cpu_count": cpu_count,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "test": test,
            "data_dir": data_dir,
            "train_dir": train_dir,
            "pretrain_dir": pretrain_dir,
            "max_len": max_len,
            "decode_strategy": decode_strategy,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)

    tokenizer = get_tokenizer(args.tokenizer_dir)
    PAD_ID = tokenizer.encoder["<|endoftext|>"]
    print("Tokenizer PAD ID:", PAD_ID)

    print("Loading Data ...")
    data, data_remove_pad = load_data(
        path=args.data_dir,
        tokenizer=tokenizer,
        PAD_ID=PAD_ID,
        field_list=["train", "dev", "test"],
        max_len=args.max_len,
    )

    if args.test is None:
        if args.extract_layer != 0:
            print("Loading Model from the extracted layers of full model")
            with open(args.model_config) as fin:
                model_config = json.load(fin)
                config = ModelConfig(**model_config)
            model = TfmrLMHeadModel(config)
            init_weights_func = get_init_weights_func(config=config)
            model.apply(init_weights_func)
            state_dict = model.state_dict()
            full_model = torch.load(args.pretrain_dir)
            ckpt = full_model.state_dict()
            mappings = [
                {},
                {"0": "0", "1": "1", "2": "2"},
                {"0": "9", "1": "10", "2": "11"},
                {"0": "0", "1": "5", "2": "11"},
            ]
            mapping = mappings[args.extract_layer]
            for key in state_dict.keys():
                # ！ TODO 这里的 key 是什么
                if key.startswith("transformer.h"):
                    name = key.split(".")
                    name[2] = mapping[name[2]]
                    name = ".".join(name)
                else:
                    name = key

                state_dict[key] = ckpt[name]

            model.load_state_dict(state_dict)

        elif args.pretrain_dir is None:
            model = TfmrLMHeadModel(config)
            init_weights_func = get_init_weights_func(config=config)
            model.apply(init_weights_func)
        else:
            if os.path.exists(args.pretrain_dir):
                print("Loading model from %s" % args.pretrain_dir)
                model = torch.load(args.pretrain_dir)
            else:
                raise RuntimeError("No such checkpoint: %s" % args.pretrain_dir)
        model.to(device)
        if using_wandb:
            wandb.watch(model)

        optimizer = optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=0
        )
        best_val_ppl = float("inf")
        best_epoch = -1
        waiting_epoch = 0

        for epoch in range(1, args.num_epochs + 1):
            start_time = time.time()
            st, ed, batch_num = 0, 0, 0
            losses = []
            while ed < len(data["train"]):
                batch_num += 1
                st_time = time.time()
                st, ed = (
                    ed,
                    (ed + args.batch_size)
                    if (ed + args.batch_size < len(data["train"]))
                    else len(data["train"]),
                )
                batched_data = torch.tensor(data["train"][st:ed]).to(device)

                optimizer.zero_grad()
                loss = model(
                    input_ids=batched_data, labels=batched_data, PAD_ID=PAD_ID
                )["loss"]
                loss.backward()
                optimizer.step()
                losses.append(loss.tolist())

                if batch_num % 10 == 0:
                    taring_loss = np.mean(losses[-100:])
                    if using_wandb:
                        wandb.log(
                            {
                                "training_epoch": epoch,
                                "batch_num": batch_num,
                                "train_loss_batch": taring_loss,
                            }
                        )
                    print(
                        "Epoch %d Batch %d, train loss %f"
                        % (epoch, batch_num, taring_loss)
                    )

            train_loss = np.mean(losses)

            val_loss, val_ppl = fast_evaluate(
                model=model,
                data=data["dev"],
                batch_size=args.batch_size,
                PAD_ID=PAD_ID,
                device=device,
            )
            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                best_epoch = epoch

                os.makedirs(args.train_dir, exist_ok=True)
                with open(
                    os.path.join(args.train_dir, "%s.tar" % args.name), "wb"
                ) as f:
                    torch.save(model, f)
                epoch_time = time.time() - start_time
                print(
                    "Epoch "
                    + str(epoch)
                    + " of "
                    + str(args.num_epochs)
                    + " took "
                    + str(epoch_time)
                    + "s"
                )
                print("  training loss:                 " + str(train_loss))
                print("  validation loss:               " + str(val_loss))
                print("  validation perplexity:         " + str(val_ppl))
                print("  best epoch:                    " + str(best_epoch))
                print("  best validation perplexity:    " + str(best_val_ppl))
                if using_wandb:
                    print("logging to wandb")
                    wandb.log(
                        {
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "val_ppl": val_ppl,
                            "best_epoch": best_epoch,
                            "best_val_ppl": best_val_ppl,
                        }
                    )
                waiting_epoch = 0
            else:
                if using_wandb:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "val_ppl": val_ppl,
                        }
                    )
                print("Validation loss: %.3f, becomes larger. Stop training." % val_ppl)
                waiting_epoch += 1
                print()
                print("waiting_epoch is " + str(waiting_epoch))
                if waiting_epoch >= args.waiting_epoch:
                    break

    else:
        #! test 直接是 path
        if os.path.exists(args.test):
            print("Loading model from %s" % args.test)
            model = torch.load(args.test)
        else:
            raise RuntimeError("No such checkpoint")
        model.to(device)
        if using_wandb:
            wandb.watch(model)
        print("Start testing.")

        print("testing batch_size is: " + str(args.batch_size))
        test_loss, test_ppl = fast_evaluate(
            model=model,
            data=data["test"],
            batch_size=args.batch_size,
            PAD_ID=PAD_ID,
            device=device,
        )
        print("        test_set, perplexity %.2f" % (test_ppl))
        result = model.inference(
            device=device,
            PAD_ID=PAD_ID,
            batch_size=args.batch_size,
            max_len=args.max_len,
            decode_strategy=args.decode_strategy,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        os.makedirs("./test_results", exist_ok=True)
        with open("./test_results/%s.txt" % args.name, "a+") as f:
            for k, output in enumerate(result):
                out = tokenizer.decode(output)
                f.write(out + "\n")
            f.write("----------------------------------------------------\n")
        eval_result = evaluate(gen_ids=result, truth_ids=data_remove_pad["test"])
        if using_wandb:
            wandb.log(
                {
                    "test_loss": test_loss,
                    "test_ppl": test_ppl,
                    "eval_result": eval_result,
                    "fw-bleu-4": eval_result["fw-bleu-4"],
                    "bw-bleu-4": eval_result["bw-bleu-4"],
                    "fw-bw-bleu-4": eval_result["fw-bw-bleu-4"],
                }
            )
        print(
            "        test_set, forward BLEU-4 %.3f, backward BLEU-4 %.3f, harmonic BLEU-4 %.3f"
            % (
                eval_result["fw-bleu-4"],
                eval_result["bw-bleu-4"],
                eval_result["fw-bw-bleu-4"],
            )
        )
