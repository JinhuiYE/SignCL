from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge  # 导入rouge库

# 读取文件的函数
def read_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

# 计算 BLEU 分数的函数
def calculate_bleu(references, candidate, n):
    weights = [1.0/n]*n + [0.0]*(4-n)  # 使总权重为 1
    return sentence_bleu(references, candidate, weights=weights, smoothing_function=SmoothingFunction().method1)

def get_results(candidates,references ):
    # 初始化rouge计算器
    rouge_calculator = Rouge()


    # 计算指标
    bleu_scores = {n: 0 for n in range(1, 5)}
    rouge_l_scores = []

    for i, candidate in enumerate(candidates):
        candidate_sentence = candidate
        reference_sentence = references[i]
        for n in range(1, 5):
            bleu_scores[n] += calculate_bleu([reference_sentence.split()], candidate.split(), n)
        # 计算ROUGE
        score = rouge_calculator.get_scores(candidate_sentence, reference_sentence)
        rouge_l_scores.append(score[0]['rouge-l']['f'])

    # 求平均
    for n in bleu_scores:
        bleu_scores[n] /= len(candidates)
    average_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)

    # 打印结果
    print("Average ROUGE-L:", average_rouge_l)
    for n in range(1, 5):
        print(f"BLEU-{n}:", bleu_scores[n])

    answer = f"&{average_rouge_l * 100:.2f} &{bleu_scores[1] * 100:.2f} &{bleu_scores[2] * 100:.2f}&{bleu_scores[3] * 100:.2f}& {bleu_scores[4] * 100:.2f}"

    print(answer)
