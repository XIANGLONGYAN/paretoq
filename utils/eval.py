from typing import Any

import argparse
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table
from .datautils import get_wikitext2, get_loaders
from lm_eval.tasks import TaskManager


from tqdm import trange, tqdm
import torch.nn.functional as F


@torch.no_grad()
def ppl_eval(model, tokenizer, tasks, seqlen=2048, cache_dir=None):
    results = {}
    task_names = tasks.split(",")

    for dataset in task_names:
        testloader = get_loaders(
            tokenizer, dataset, seqlen=seqlen, cache_dir=cache_dir
        )
        testenc = testloader if "c4" in dataset else testloader.input_ids
        nsamples = testenc.numel() // seqlen
        use_cache = model.config.use_cache
        model.config.use_cache = False
        model.eval()

        if hasattr(model, "lm_head") and isinstance(model.lm_head, torch.nn.Linear):
            classifier = model.lm_head
        elif hasattr(model.model, "lm_head"):
            classifier = None
        elif hasattr(model, "output"):
            classifier = model.output
        else:
            raise NotImplementedError

        nlls = []
        for i in tqdm(range(nsamples)):
            batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(model.device)
            outputs = model.model(batch)
            if classifier is not None:
                hidden_states = outputs[0]
                logits = classifier(hidden_states.to(classifier.weight.dtype))
            else:
                logits = outputs[0]
            shift_logits = logits[:, :-1, :]
            shift_labels = testenc[:, (i * seqlen) : ((i + 1) * seqlen)][:, 1:].to(
                shift_logits.device
            )
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)

        results[dataset] = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen)).item()
    model.config.use_cache = use_cache

    return results

def run_evaluation(model, tokenizer, tasks, batch_size="auto", eval_ppl=False, eval_lm_eval=False,):
    
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"WARNING: {name} contains NaN/Inf!")

    """运行 ppl 评估"""
    if eval_ppl:
        ppl_tasks = "wikitext2,c4"
        ppl_results = ppl_eval(model, tokenizer, tasks=ppl_tasks, seqlen=2048, cache_dir=None)
        for dataset, ppl in ppl_results.items():
            print(f"{dataset} perplexity: {ppl:.2f}")

    """运行 lm_eval 评估"""
    if eval_lm_eval:
        # 需要 apply_chat_template 的任务集合

        chat_template_tasks = {"mmlu_cot_llama", "gsm8k_llama", "mmlu_llama"}  # 根据需要添加,        

        
        # 将任务分成两组
        tasks_with_chat = [t for t in tasks if t in chat_template_tasks]
        tasks_without_chat = [t for t in tasks if t not in chat_template_tasks]

        lm_obj = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
        task_manager = TaskManager()

        results1 = {}
        results2 = {}

        # 第一批：不需要 chat template 的任务
        if tasks_without_chat:
            print(f"\n开始评估任务 (无 chat template): {tasks_without_chat}")
            results1 = {}

            for task_name in tasks_without_chat:
                print(task_name)
                task_results = lm_eval.simple_evaluate( # call simple_evaluate
                    model=lm_obj,
                    tasks=[task_name],
                    num_fewshot=0,
                    batch_size=batch_size,
                    task_manager=task_manager,
                )["results"]

                results1.update(task_results)
                print(make_table({"results": task_results, "versions": {}, "n-shot": {}, "higher_is_better": {}}))

        # 第二批：需要 chat template 的任务
        if tasks_with_chat:
            print(f"\n开始评估任务 (有 chat template): {tasks_with_chat}")

            results2 = {}

            for task_name in tasks_with_chat:
                print(task_name)
                task_results = lm_eval.simple_evaluate( # call simple_evaluate
                    model=lm_obj,
                    tasks=[task_name],
                    batch_size=batch_size,
                    task_manager=task_manager,
                    apply_chat_template=True,
                    fewshot_as_multiturn=True,
                )["results"]

                results2.update(task_results)
                print(make_table({"results": task_results, "versions": {}, "n-shot": {}, "higher_is_better": {}}))

        all_results = {}
        if tasks_without_chat:
            all_results.update(results1)
        if tasks_with_chat:
            all_results.update(results2)
        
        if all_results:
            print("\n" + "="*60)
            print("Summary:")
            total_acc = 0
            count = 0
            for task, metrics in all_results.items():
                acc = metrics.get('acc_norm,none', metrics.get('acc,none', None))
                if acc is not None:
                    print(f"  {task}: {acc*100:.2f}%")
                    total_acc += acc
                    count += 1
            if count > 0:
                print(f"  Average Acc: {total_acc/count*100:.2f}%")
            print("="*60)

        if eval_ppl:
            for dataset, ppl in ppl_results.items():
                print(f"{dataset} perplexity: {ppl:.2f}")

        return