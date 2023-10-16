from ambigqa_evaluate_script import get_exact_match, get_f1, normalize_answer
import numpy as np
from pycocoevalcap.bleu.bleu import Bleu
import evaluate
import difflib
from CQDataset import parse_clarification_question
import json
from scipy.optimize import linear_sum_assignment
from bert_score import score

def CQA_evaluate(golds, predictions, gold_option, pred_option, dq_type, question_idx_list, dev_len):
    
    grouped_predictions=[[] for i in range(np.max(question_idx_list)+1)]
    avg_ans=len(predictions)/len(grouped_predictions)
    uniq_ans=len(set([normalize_answer(d) for d in predictions]))/len(grouped_predictions)
    for question_idx, pred in zip(question_idx_list, predictions):
        grouped_predictions[question_idx].append(normalize_answer(pred))

    grouped_golds=[[[normalize_answer(ans) for ans in answer_set] for answer_set in gold] for gold in golds]
    if 'gold' in dq_type:  # When we know the correspondance btw preds & gold answers
        golds=[[normalize_answer(ans) for ans in answer_set] for gold in golds for answer_set in gold]
        ems=[1 if normalize_answer(pred) in golds[i] else 0 for i, pred in enumerate(predictions)]    
        em = np.mean(ems)
        with open('ems_for_{}.json'.format(dq_type), 'w') as f:
            json.dump(ems,f)
        
        with open('predictions_for_{}.json'.format(dq_type), 'w') as f:
            json.dump(grouped_predictions, f)
        with open('golds_for_{}.json'.format(dq_type), 'w') as f:
            json.dump(grouped_golds, f)
            
        p, r, f, p_score_list, r_score_list = compute_scores(grouped_golds, grouped_predictions, score_fn=count_overlap_list)
      
        
           
        return {'EM':em, 'Partial_Match_Precision':p, 'Partial_Match_Recall': r, 'Partial_Match_F1': f, 'Average_number_of_gen_answers': avg_ans, 'Average_number_of_unique_answers': uniq_ans}, p_score_list, r_score_list
    else:                  # When we do not know the correspondance btw preds & gold answers
        assert(gold_option is not None)
        assert(pred_option is not None)
        
        p, r, f, p_score_list, r_score_list = compute_scores(grouped_golds, grouped_predictions, score_fn=count_overlap_list)
        
        return {'Partial_Match_Precision':p, 'Partial_Match_Recall':r, 'Partial_Match_F1':f, 'Average_number_of_gen_answers': avg_ans, 'Average_number_of_unique_answers': uniq_ans}, p_score_list, r_score_list

def CQG_evaluate (golds, predictions, dev_len):
    bleu=evaluate.load("bleu")
    cq_bleu=bleu.compute(predictions=predictions, references=golds)
    
    #BERT_SCORE
    _, _, bertscore= score(golds, predictions, lang='en', verbose = True)
    
    #Category_Bleu
    gold_category=[parse_clarification_question(g)[0] for g in golds]
    pred_category=[parse_clarification_question(p)[0] for p in predictions]
    category_em=np.mean([1 if pred_category[idx]==gold_category[idx] else 0 for idx in range(len(predictions))])
    category_bleu=bleu.compute(predictions=pred_category, references=gold_category, max_order=1)
    
    #Options Partial Match
    gold_option=[parse_clarification_question(g)[1] for g in golds]
    pred_option=[parse_clarification_question(p)[1] for p in predictions]
    opt_num=0
    for p in pred_option:
        opt_num+=len(p)
    p, r, f, p_score_list, r_score_list = compute_scores(gold_option, pred_option)
    avg_opts=opt_num/dev_len
    return {'CQ_BLEU':cq_bleu['bleu'], 'CQ_BERTSCORE':np.mean(bertscore.tolist()), 'Category_EM':category_em, 'Category_BLEU':category_bleu['bleu'],
            'Option_precision':p,'Option_recall':r, 'Option_f1':f, 'Average_number_of_options': avg_opts}


def count_overlap(gold: set, pred: set):
    """Count the overlap of the gold answer and the predicted answer.
    :param gold: Set of gold answers
    :param pred: Set of predicted answers
    """
    # Correct no answer prediction
    if len(gold) == 0 and (len(pred) == 0 or pred == {""}):
        return 1, 1

    # Incorrect no answer prediction
    elif len(gold) == 0 or (len(pred) == 0 or pred == {""}):
        return 0, 0

    # NOTE: Since it is possible to return multiple spans it is not clear which spans from pred should be compared to
    #       each span in gold. So all are compared and the highest precision and recall are taken.
    p_scores = np.zeros((len(gold), len(pred)))
    r_scores = np.zeros((len(gold), len(pred)))
    for i, gold_str in enumerate(gold):
        for j, pred_str in enumerate(pred):
            seq_matcher = difflib.SequenceMatcher(None, gold_str, pred_str)
            _, _, longest_len = seq_matcher.find_longest_match(0, len(gold_str), 0, len(pred_str))
            p_scores[i][j] = longest_len/len(pred_str) if longest_len > 0 else 0
            r_scores[i][j] = longest_len/len(gold_str) if longest_len > 0 else 0

    prec_row_ind, prec_col_ind = linear_sum_assignment(p_scores, maximize=True)
    max_prec= p_scores[prec_row_ind, prec_col_ind].sum()

    rec_row_ind, rec_col_ind = linear_sum_assignment(r_scores, maximize=True)
    max_rec = r_scores[rec_row_ind, rec_col_ind].sum()
    
    return max_prec, max_rec



def count_overlap_list(gold: set, pred: set): # gold: list of list of list, pred: list of list
    """Count the overlap of the gold answer and the predicted answer.
    :param gold: Set of gold answers
    :param pred: Set of predicted answers
    """
    # Correct no answer prediction
    if len(gold) == 0 and (len(pred) == 0 or pred == {""}):
        return 1, 1

    # Incorrect no answer prediction
    elif len(gold) == 0 or (len(pred) == 0 or pred == {""}):
        return 0, 0

    # NOTE: Since it is possible to return multiple spans it is not clear which spans from pred should be compared to
    #       each span in gold. So all are compared and the highest precision and recall are taken.
    p_scores = np.zeros((len(gold), len(pred)))
    r_scores = np.zeros((len(gold), len(pred)))
    for i, gold_set in enumerate(gold):
        for j, pred_str in enumerate(pred):
            longest_idx ,longest_len=overlap_btw_str_list(pred_str, gold_set)
            p_scores[i][j] = longest_len/len(pred_str) if longest_len > 0 else 0
            r_scores[i][j] = longest_len/len(gold_set[longest_idx]) if longest_len > 0 else 0

    prec_row_ind, prec_col_ind = linear_sum_assignment(p_scores, maximize=True)
    max_prec= p_scores[prec_row_ind, prec_col_ind].sum()

    rec_row_ind, rec_col_ind = linear_sum_assignment(r_scores, maximize=True)
    max_rec = r_scores[rec_row_ind, rec_col_ind].sum()
    
    # p_score = sum(np.max(p_scores, axis=0))
    # r_score = sum(np.max(r_scores, axis=1))

    return max_prec, max_rec

def overlap_btw_str_list(pred_str, gold_set):
    longest_len_list=[]
    for i, gold_str in enumerate(gold_set):
        seq_matcher = difflib.SequenceMatcher(None, gold_str, pred_str)
        _, _, longest_len= seq_matcher.find_longest_match(0, len(gold_str), 0, len(pred_str))
        longest_len_list.append(longest_len)
    
    return np.argmax(longest_len_list), np.max(longest_len_list)

def compute_scores(golds, preds, average: str = 'micro', score_fn=count_overlap):
    """Compute precision, recall and exact match (or f1) metrics.
    :param golds: dictionary of gold XX
    :param preds: dictionary of predictions
    :param eval_type: Evaluation type. Can be either "em" or "overlap".
    """
    nb_gold = 0
    nb_pred = 0
    nb_correct = 0
    nb_correct_p = 0
    nb_correct_r = 0
    assert(len(golds)==461)
    for idx, (gold, pred) in enumerate(zip(golds, preds)):
        p_score_list=[]
        r_score_list=[]
        nb_gold += max(len(gold), 1)
        nb_pred += max(len(pred), 1)
    
        p_score, r_score = score_fn(gold, pred)
        nb_correct_p += p_score
        nb_correct_r += r_score
        p_score_list.append(p_score)
        r_score_list.append(r_score)
        
        # if idx not in correct_pred:
        #     nb_gold += max(len(gold), 1)
        #     nb_pred += max(len(pred), 1)
        
        #     p_score, r_score = score_fn(gold, pred)
        #     nb_correct_p += p_score
        #     nb_correct_r += r_score
        # else:
        #     nb_gold += max(len(gold), 1)
        #     nb_pred += 1 # Ambiguous but model predicted as non-ambiguous model prediction =1
    
    p = nb_correct_p / nb_pred if nb_pred > 0 else 0
    r = nb_correct_r / nb_gold if nb_gold > 0 else 0

    f = 2 * p * r / (p + r) if p + r > 0 else 0

    return p, r, f, p_score_list, r_score_list

#Bert prediction
#correct_pred=[0, 1, 5, 6, 10, 11, 12, 13, 14, 18, 19, 24, 25, 26, 29, 32, 33, 35, 38, 39, 43, 44, 47, 51, 52, 53, 54, 56, 58, 59, 60, 61, 62, 63, 65, 66, 67, 68, 69, 71, 72, 76, 79, 81, 82, 83, 85, 86, 87, 88, 89, 90, 93, 94, 96, 98, 101, 102, 103, 104, 105, 108, 110, 111, 112, 113, 114, 115, 117, 119, 122, 123, 127, 128, 130, 131, 132, 133, 136, 137, 139, 142, 144, 145, 146, 149, 151, 152, 160, 161, 162, 163, 164, 165, 166, 168, 169, 170, 173, 175, 176, 177, 178, 180, 182, 183, 185, 186, 187, 188, 189, 192, 193, 194, 196, 197, 198, 199, 201, 202, 203, 204, 205, 207, 209, 211, 212, 214, 216, 217, 223, 231, 232, 234, 238, 239, 240, 241, 242, 243, 245, 246, 247, 250, 252, 253, 254, 256, 257, 259, 261, 262, 264, 265, 267, 268, 269, 270, 273, 275, 276, 277, 278, 279, 281, 284, 285, 286, 287, 288, 289, 292, 293, 296, 297, 299, 301, 302, 303, 304, 305, 307, 308, 310, 312, 313, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 328, 329, 332, 333, 334, 336, 337, 339, 340, 341, 342, 344, 345, 346, 347, 348, 350, 352, 353, 354, 355, 358, 359, 360, 364, 365, 366, 368, 370, 371, 372, 374, 375, 377, 380, 382, 384, 386, 388, 390, 391, 392, 393, 398, 400, 401, 402, 405, 407, 409, 410, 414, 415, 416, 418, 419, 420, 421, 422, 424, 425, 427, 428, 429, 431, 432, 433, 436, 438, 439, 440, 441, 442, 443, 444, 445, 447, 448, 449, 450, 453, 455, 458, 459]

# SPANSEQGEN prediction
#correct_pred=[0, 6, 7, 9, 13, 17, 22, 24, 26, 41, 42, 48, 57, 60, 61, 66, 68, 69, 80, 82, 83, 84, 90, 91, 93, 96, 99, 110, 116, 124, 127, 129, 131, 140, 147, 154, 158, 160, 163, 167, 168, 171, 175, 177, 183, 191, 193, 194, 197, 204, 205, 207, 212, 214, 228, 231, 237, 240, 245, 249, 250, 251, 253, 263, 265, 269, 270, 275, 276, 281, 283, 292, 293, 301, 304, 305, 306, 307, 308, 310, 318, 327, 328, 331, 336, 348, 350, 358, 362, 366, 370, 374, 379, 382, 386, 391, 398, 401, 402, 405, 416, 424, 425, 428, 429, 436, 439, 441, 447, 453, 456, 458]

# Both wrong
#wrong_pred=[2, 3, 4, 8, 15, 16, 20, 21, 23, 27, 28, 30, 31, 34, 36, 37, 40, 45, 46, 49, 50, 55, 64, 70, 73, 74, 75, 77, 78, 92, 95, 97, 100, 106, 107, 109, 118, 120, 121, 125, 126, 134, 135, 138, 141, 143, 148, 150, 153, 155, 156, 157, 159, 172, 174, 179, 181, 184, 190, 195, 200, 206, 208, 210, 213, 215, 218, 219, 220, 221, 222, 224, 225, 226, 227, 229, 230, 233, 235, 236, 244, 248, 255, 258, 260, 266, 271, 272, 274, 280, 282, 290, 291, 294, 295, 298, 300, 309, 311, 314, 325, 326, 330, 335, 338, 343, 349, 351, 356, 357, 361, 363, 367, 369, 373, 376, 378, 381, 383, 385, 387, 389, 394, 395, 396, 397, 399, 403, 404, 406, 408, 411, 412, 413, 417, 423, 426, 430, 434, 435, 437, 446, 451, 452, 454, 457, 460]