import time
import json
import os
from evaluation.lm_eval import evaluator
from evaluation.lm_eval.models.huggingface import HFLM
from evaluation.lm_eval.tasks import initialize_tasks

LM_EVAL_TASK_KWARGS_DICT = {

    "hellaswag": {"task": "hellaswag", "num_fewshot": 0, "batch_size": 128, "metric": "acc_norm"},
    # "lambada_openai": {"task": "lambada_openai", "num_fewshot": 0, "batch_size": 128, "metric": "acc"},
    # "piqa": {"task": "piqa", "num_fewshot": 0, "batch_size": 128, "metric": "acc"},
}

def eval_zeroshot(model, tokenizer, result_save_path, report_both_metrics: bool = True):
    begin = time.time()
    print(f"Start zero-shot evaluation")
    save_file_path = os.path.join(result_save_path, "eval_result.json")

    all_metrics = {}
    if os.path.exists(save_file_path):
        with open(save_file_path, 'r') as file:
            all_metrics = json.load(file)

    for task_kwargs in LM_EVAL_TASK_KWARGS_DICT.values():
        print(f"Evaluating task: {task_kwargs['task']}")
        task_name = task_kwargs["task"]
        lm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=task_kwargs["batch_size"],
        )
        initialize_tasks(verbosity="ERROR")
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=task_name,
            num_fewshot=task_kwargs["num_fewshot"],
            batch_size=task_kwargs["batch_size"],
            log_samples=False,
        )
        metric = task_kwargs["metric"]
        task_results = results.get("results", {}).get(task_name, {})
        for key, value in task_results.items():
            if isinstance(key, tuple):
                mname = key[0]
            else:
                mname = key.split(",")[0]
            if mname == metric:
                all_metrics[f"{task_name}_{metric}"] = value

        if report_both_metrics and task_name == "hellaswag":
            for m in ["acc", "acc_norm"]:
                for key, value in task_results.items():
                    if isinstance(key, tuple):
                        mname = key[0]
                    else:
                        mname = key.split(",")[0]
                    if mname == m:
                        all_metrics[f"{task_name}_{m}"] = value

        with open(save_file_path, 'w') as file:
            json.dump(all_metrics, file, indent=4)
    end = time.time()
    print(">>>>> Results <<<<<")
    # average = sum(v for v in all_metrics.values()) / len(all_metrics)
    # all_metrics["average"] = average
    print(f"Metrics: {all_metrics}")
    print(f"Evaluation time: {end - begin:.2f}s")
    

    
