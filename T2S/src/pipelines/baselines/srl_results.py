import json
import os
import numpy as np

if __name__ == '__main__':
    srl_dir = "../../../resultfiles/baselines/SRL/all_topics/"

    srl_list = []
    for subdir, dirs, files in os.walk(srl_dir):
        print(subdir)
        for _dir in dirs:
            # print(_dir)
            with open(subdir+_dir+"/srl_coref.json", encoding="utf-8") as f:
                data = json.load(f)
            srl_list.append(data)

    print(sum(srl_metrics["srl_metrics"]["global"]["rouge1_precision"] for srl_metrics in srl_list) / len(srl_list))

    precision_sum, recall_sum, f1_sum = 0, 0, 0
    max_p, max_r, max_f = 0, 0, 0
    total_cases = len(srl_list)
    for srl in srl_list:
        verbs_metrics = srl["srl_metrics"]["per_verb"]

        best_verb_precisions, best_verb_recall, best_verb_f1 = [], [], []
        if len(verbs_metrics) != 0:
            for verb in verbs_metrics:
                best_verb_precisions.append(verbs_metrics[verb]["best_frames"]["best_precision"]["rouge1_precision"])
                best_verb_recall.append(verbs_metrics[verb]["best_frames"]["best_recall"]["rouge1_recall"])
                best_verb_f1.append(verbs_metrics[verb]["best_frames"]["best_f1"]["rouge1_f1"])

            precision_sum += np.array(best_verb_precisions).mean()
            recall_sum += np.array(best_verb_recall).mean()
            f1_sum += np.array(best_verb_f1).mean()

            max_p += max(best_verb_precisions)
            max_r += max(best_verb_recall)
            max_f += max(best_verb_f1)
        else:
            total_cases -= 1

    print("\nGlobal SRL results - mean of each verbs best frame:\n")
    print("precision -", precision_sum / total_cases)
    print("recall -", recall_sum / total_cases)
    print("f1 -", f1_sum / total_cases)

    print("\nGlobal SRL results - best frame for the best verb:\n")
    print("precision -", max_p / total_cases)
    print("recall -", max_r / total_cases)
    print("f1 -", max_f / total_cases)
