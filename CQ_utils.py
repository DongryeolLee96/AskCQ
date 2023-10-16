import os
import json
import gzip
import re
import sys
import pickle as pkl
import string
import numpy as np
import QGData
from termcolor import colored
from tqdm import tqdm
from collections import Counter, defaultdict
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pathlib import Path
from copy import deepcopy
import random
import numpy as np
from datasets import Dataset
import spacy
NLP = spacy.load('en_core_web_sm')


def write_cq_in_NL(ap, options, mark_ap=False):
    if ap is None:
        if mark_ap:
            return f"Could you be more specific--" + list_options_in_NL(options) + "?"
        return f"Could you be more specific--" + list_options_in_NL(options) + "?"
    else:
        if mark_ap:
            return f"Could you clarify \'<strong><ins>{ap}</ins></strong>\'--" + list_options_in_NL(options) + "?"
        return f"Could you clarify \'{ap}\'--" + list_options_in_NL(options) + "?"


def list_options_in_NL(options):
    options = ["\'" + option + "\'" for option in options]
    if len(options) > 2:
        return ", ".join(options[:-1]) + f", or {options[-1]}"
    elif len(options) <= 2:
        return " or ".join(options)
    else:
        return ""


def list_answer_set_in_NL(answer_set):
    if len(answer_set) > 1:
        primary = answer_set[0]
        others = ", ".join(answer_set[1:])
        ans = f"{primary} (={others})"
    else:
        ans = answer_set[0]
    return ans


# PRINTING FUNCTIONS
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def blue(w):
    return colored(w, color="blue")

def cyan(w):
    return colored(w, color="cyan")

def white(w):
    return colored(w, color="white")

def red(w):
    return colored(w, color="red")

def on_yellow(w):
    return colored(w, on_color="on_yellow")

def on_green(w):
    return colored(w, on_color="on_green")

def highlight_part(text, root, always=False):
    mark = on_yellow
    if "\t" in root:
        root = root.split("\t")[1]
        mark = on_green
    root_span = re.search(re.escape(root), text)
    if root_span:
        s, e = root_span.start(), root_span.end()
        res = text[:s]
        if always or e-s != len(text):
            res += mark("".join(text[idx] for idx in range(s, e)))
        else:
            res += "".join(text[idx] for idx in range(s, e))
        res += text[e:]
        return res
    return text


def remove_redundant_pairs(dqs, answers):
    redundancy = [0] * len(dqs)
    for idx1, ans1 in enumerate(answers):
        ans1_ = ans1[0]
        for idx2, ans2 in enumerate(answers):
            if idx1 >= idx2:
                continue
            ans2_ = ans2[0]
            if redundancy[idx2] == 1:
                continue
            if ans1 == ans2:
                redundancy[idx2] = 1
                continue
            if is_subseq(ans2_.lower().split(), ans1_.lower().split()):
                redundancy[idx2] = 1
                continue
            elif is_subseq(ans1_.lower().split(), ans2_.lower().split()):
                redundancy[idx1] = 1
                continue
    dqs_ = [dq for idx, dq in enumerate(dqs) if redundancy[idx] == 0]
    answers_ = [ans for idx, ans in enumerate(answers) if redundancy[idx] == 0]
    return dqs_, answers_


# ALGORITHMIC FUNCTIONS
def lcs(X, Y):
    # find the length of the strings
    m = len(X)
    n = len(Y)
    # declaring the array for storing the dp values
    ids_x = [[[] for _ in range(n + 1)] for i in range(m + 1)]
    ids_y = [[[] for _ in range(n + 1)] for i in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                continue
            elif X[i - 1] == Y[j - 1]:
                ids_x[i][j] = ids_x[i - 1][j - 1] + [i - 1]
                ids_y[i][j] = ids_y[i - 1][j - 1] + [j - 1]
            else:
                if len(ids_y[i - 1][j]) > len(ids_y[i][j - 1]):
                    ids_x[i][j] = ids_x[i - 1][j]
                    ids_y[i][j] = ids_y[i - 1][j]
                else:
                    ids_x[i][j] = ids_x[i][j - 1]
                    ids_y[i][j] = ids_y[i][j - 1]
    return ids_x[-1][-1], ids_y[-1][-1]  # indices in Y.


def get_shell_of_span(ids, span):
    id2pos = {id: idx for idx, id in enumerate(ids)}
    if span[-1] < ids[0]:
        lb_pos = -1
    for lb in range(span[0], ids[0]-1, -1):
        if lb in id2pos:
            lb_pos = id2pos[lb]
            break

    if span[0] > ids[-1]:
        ub_pos = 100
    for ub in range(span[-1], ids[-1]+1):
        if ub in id2pos:
            ub_pos = id2pos[ub]
            break
    return lb_pos, ub_pos


def is_subseq(x, y):  # x <= y
    it = iter(y)
    return all(any(c == ch for c in it) for ch in x)


def get_span_pairs(ids_aq, spans_aq, ids_dq, spans_dq):
    span_pairs = []
    for idx, span_dq in enumerate(spans_dq):
        shell_dq = get_shell_of_span(ids_dq, span_dq)
        for span_aq in spans_aq:
            shell_aq = get_shell_of_span(ids_aq, span_aq)
            if shell_aq == shell_dq:
                span_pairs.append((span_aq, span_dq))
        if len(span_pairs) < idx + 1:
            span_pairs.append((None, span_dq))
    return span_pairs


def is_textual_subspan(x, y):  # "the weather" <= "What is the weather"
    if x==y:
        return True
    if len(re.findall(f" {x}$", y)) > 0:
        return True
    if len(re.findall(f"^{x} ", y)) > 0:
        return True
    if len(re.findall(f" {x} ", y)) > 0:
        return True
    return False


def cleanse_list_by_merging(wlist):
    # [[2, 3, 4, 5], [2, 3], [5, 6]] -> [[2, 3, 4, 5], [5, 6]]
    new_words = sorted(wlist, key=len, reverse=True)
    final_results = [a for i, a in enumerate(new_words) if not any(set(a).issubset(set(c)) for c in new_words[:i])]
    return final_results


def merge_meaningful_adjacency(ids, doc):
    spans = merge_adjacency(ids)
    return [span for span in spans if is_meaningful_span(doc, span)]


def merge_adjacency(ids):
    if len(ids) == 0:
        return []
    prev = ids[0]
    span = [ids[0]]
    spans = []
    for idx in ids[1:]:
        if idx == prev + 1:
            span.append(idx)
        else:
            spans.append(span)
            span = [idx]
        prev = idx
    if len(span) > 0:
        spans.append(span)
    return spans


def substitute_propns(sen, propns, reverse=False):
    if reverse:
        for idx, propn in enumerate(propns):
            sen = re.sub(f" propn{idx}$| PROPN{idx}$", f" {propn}", sen)
            sen = re.sub(f"^propn{idx} |^PROPN{idx} ", f"{propn} ", sen)
            sen = re.sub(f"^propn{idx}$|^PROPN{idx}$", f"{propn}", sen)
            sen = re.sub(f" propn{idx} | PROPN{idx} ", f" {propn} ", sen)
        return sen
    for idx, propn in enumerate(propns):
        sen = re.sub(f" {propn}$", f" PROPN{idx}", sen)
        sen = re.sub(f"^{propn} ", f"PROPN{idx} ", sen)
        sen = re.sub(f" {propn} ", f" PROPN{idx} ", sen)
        sen = re.sub(f"^{propn}$", f"PROPN{idx}", sen)
    return sen


def find_the_most_similar_title(viewed_doc_title, candidate_titles):
    a = viewed_doc_title.split(" ")
    max_score = -1
    for candidate in candidate_titles:
        b = candidate.split(" ")
        aa, bb = lcs(a, b)
        score1 = len(aa) / len(a)
        score2 = len(aa) / len(b)
        if score1 * score2 == 1:
            return candidate
        score = score1 * 10 + len(aa) / len(b)
#         print(score, candidate)
        if score > max_score:
            res = candidate
            max_score = score
    return res


# FUNCTIONS WITH DOC INPUTS
def get_tokens(doc):
    return [token.text for token in doc]


def get_root(doc, return_idx=False, allow_aux_root=True):
    idx, token = [(idx, token) for idx, token in enumerate(doc) if token.dep_ == "ROOT"][0]
    if not allow_aux_root and token.pos_ == "AUX":  # is, are, did, ...
        span = [i for i in range(len(doc)) if i != idx]
        if len(span) == 0:
            if return_idx:
                return idx
            return token.text
        chunk = span_to_chunk(doc, span)
        return get_root(NLP(chunk), return_idx)
    if return_idx:
        return idx
    return token.text


def find_leftmost_noun(doc, idx_from=0):
    # cp_leaf = dq_doc[cp_root_idx+1]
    # if cp_leaf.pos_ == "DET":
    #     head_of_cp_leaf = dq_doc[cp_root_idx+1].head.text
    # else:
    #     head_of_cp_leaf = dq_doc[cp_root_idx+1].text
    for token in doc[idx_from:]:
        if token.pos_ in ["NOUN", "PROPN", "PRON"]:
            return token
    return doc[idx_from]


def highlight_difference(ambig_doc, unambig_doc):
    X = get_tokens(ambig_doc)
    Y = get_tokens(unambig_doc)
    _, ids = lcs(X, Y)
    res = ""
    for idx, w in enumerate(Y):
        #     for idx, w in enumerate(unambig.split(" ")):
        w = w.replace("propn", "PROPN")
        if "PROPN" in w:
            res += " " + colored(w, 'blue')
        elif idx not in ids:
            res += " " + colored(w, 'red')
        else:
            res += " " + w
    print(res)


def get_spans_of_noun_chunks(doc):
    # doc = nlp("Autonomous cars shift insurance liability toward manufacturers")
    # https://spacy.io/usage/linguistic-features
    span_by_noun_chunks = [[i for i in range(chunk.start, chunk.end)] for chunk in doc.noun_chunks]
    return span_by_noun_chunks


def is_meaningful_span(doc, span):
    if len(span) == 0:
        return False
    if all([doc[i].pos_ in ["ADP", "AUX", "DET", "PRON", "PUNCT"] for i in span]):
        # Exclude ADP: of, in, under, etc...
        # Exclude AUX: should, could, has, )
        return False
    return True


def span_to_chunk(doc, span, specials={}):
    noun = ''.join([specials.get(idx, doc[idx].text_with_ws) for idx in span]).strip()
    return noun


def find_cand_with_propn_inside_dp(roots_of_candidates, dp):
    # [ pro ... pn ]
    for k, v in roots_of_candidates.items():
        if "propn" in k:
            propn = k.split("\t")[1]
            if is_subseq(propn.split(), dp.split()):
                return k
    return None


def find_cand_substituted_by_dp(roots_of_cands, dp_aq_doc, dp_dq_doc):
    if dp_aq_doc:
        if len(dp_aq_doc) == len(dp_dq_doc) and all(t1.lemma_==t2.lemma_ for t1, t2 in zip(dp_aq_doc, dp_dq_doc)):
            return None
        print(f"\t ({dp_dq_doc} " + blue("<=>") + f" {dp_aq_doc})")
        dp_aq_root = get_root(dp_aq_doc, allow_aux_root=False)
        dp_aq_root = select_root_similar_to_tail(dp_aq_root, roots_of_cands)
        return dp_aq_root
    return None


def scoring(dp_span_aq, aq_doc, dp_span_dq, dq_doc, idx_dq, roots_of_cands, scores):
    # print("dp_span_aq:", dp_span_aq)
    # print("aq_doc:", aq_doc)
    # print("dp_span_dq:", dp_span_dq)
    # print("dq_doc:", dq_doc)
    # print("roots_of_cands:", roots_of_cands)
    if dp_span_aq:
        dp_aq = span_to_chunk(aq_doc, dp_span_aq)
        dp_aq_doc = NLP(dp_aq)
    else:
        dp_aq, dp_aq_doc = None, None
    dp_dq = span_to_chunk(dq_doc, dp_span_dq)
    dp_dq_doc = NLP(dp_dq)

    cand_with_propn_inside_dp = find_cand_with_propn_inside_dp(roots_of_cands, dp_dq)
    cand_substituted_by_dp = find_cand_substituted_by_dp(roots_of_cands, dp_aq_doc, dp_dq_doc)
    dp_text = dp_dq
    if cand_with_propn_inside_dp:
        case = "[pro ... pn]"
        score = 100
        pred = blue("CONTAINS")
        tail = cand_with_propn_inside_dp
    elif dp_dq_doc[0].pos_ == "PART":
        case = "['s ...]"
        score = 100
        tail = dq_doc[dp_span_dq[0]].head.text
        pred = blue("IS AN ASPECT OF")
    elif dp_dq_doc[0].text in [":", "("] and dp_span_dq[0] > 0:
        case = "[: ...] | [( ...]"
        score = 100
        tail = dq_doc[dp_span_dq[0] - 1].text
        pred = blue("MODIFIES")
    elif cand_substituted_by_dp:  # c.e.g., 26
        case = "parallel [...] exists"
        score = 1
        pred = blue("SUBSTITUTES")
        tail = cand_substituted_by_dp
    elif dp_span_dq[-1] < len(dq_doc) - 1 and dp_dq_doc[-1].pos_ == "PART":
        case = "[... 's]"
        score = 100
        tail = find_leftmost_noun(dq_doc, dp_span_dq[-1] + 1).text
        pred = blue("HAS")
    elif dp_span_dq[-1] < len(dq_doc) - 1 and "propn" in dq_doc[dp_span_dq[-1] + 1].text and dp_dq[-1] != ",":
        case = "[...] propn, ... not ended with comma"  # c.e.g.: 506
        score = 100
        tail = find_leftmost_noun(dq_doc, dp_span_dq[-1] + 1).text
        pred = blue("MODIFIES")
    elif dp_span_dq[0] > 0 and "propn" in dq_doc[dp_span_dq[0] - 1].text and dq_doc[dp_span_dq[0]].pos_ not in ["ADP", "PUNCT"]:
        case = "propn [...], ... not started with ADP"  # ADP: in, of, by, during, ... , PUNCT: ,
        score = 100
        tail = dq_doc[dp_span_dq[0] - 1].text
        pred = blue("IS MODIFIED BY")
    elif dp_dq_doc[-1].pos_ in ["ADP"] and dp_span_dq[-1] < len(dq_doc) - 1:
        case = "[... of]"
        tail = find_leftmost_noun(dq_doc, dp_span_dq[-1] + 1).text
        score = 1
        pred = blue("IS MODIFIED BY")
    else:
        case = "find head of [...] by spacy"
        dp_root_idx = get_root(dp_dq_doc, return_idx=True) + dp_span_dq[0]
        root = dq_doc[dp_root_idx].text
        tail = dq_doc[dp_root_idx].head.text
        score = 1
        dp_text = highlight_part(dp_dq, root)
        pred = blue("HEADING TO")

    selected_root = select_root_similar_to_tail(tail, roots_of_cands)
    if selected_root:
        scores[(dp_dq, selected_root)].append((idx_dq, score, case))

    if selected_root and "\t" in selected_root:
        selected_root = selected_root.split("\t")[1]
    print("\t", dp_text, pred, tail, blue(" ### "), selected_root, white(f"+{score}"))
    return dp_dq


def summarize_score(scores, roots_of_cands, cq_options, aq_root, aq_wh_word, preferences=[]):
    if aq_root in roots_of_cands:
        preferences = [aq_root] + preferences
    final_score = {k: 0.0 for k in roots_of_cands.keys()}
    reasons = Counter()
    for idx, preference in enumerate(preferences):
        final_score[preference] += 0.1 * 1.0/(idx+1)
    for ht, vs in scores.items():
        if len(vs) == len(cq_options):  # Every dq modifies the tail with the same head. => NO NEED TO CLARIFY
            continue
        head, tail = ht
        for v in vs:
            idx_dq, score, case = v
            final_score[tail] += score
            reasons[case] += 1
    # print(red("score: "), final_score)
    root_of_ap = max(final_score, key=final_score.get)  # ap: (chosen) ambig part
    contribution = get_contribution(scores, root_of_ap)
    ap = roots_of_cands[root_of_ap]

    # if aq_wh_word and ap == aq_root:
    if any(get_leading_wh(NLP(dp)) for dp in cq_options) and ap == aq_root:  # c.e.g. 3002
        print(f"Exception: {aq_wh_word}")
        reasons["WH-exception"] += 1
        contribution = {}
        contribution["WH-exception"] = 1
        ap = aq_wh_word
    return ap, final_score, reasons, contribution


def get_contribution(scores, root_of_ap):
    effective_scores = Counter()
    for ht, vs in scores.items():
        head, tail = ht
        if tail == root_of_ap:
            for v in vs:
                idx_dq, score, case = v
                effective_scores[case] += score
    total = sum([v for k, v in effective_scores.items()])
    contribution = {case: score/total for case, score in effective_scores.items()}
    return contribution


def select_root_similar_to_tail(tail, roots_of_candidates):
    if tail:
        for root in roots_of_candidates.keys():
            if tail in root.lower():
                return root
        for root, cand in roots_of_candidates.items():  # For capturing propn
            if tail in cand:
                return root
    return None


def generate_machine_cq(ex):
    with HiddenPrints():
        ex_ = add_candidates(ex)
        ap, cq_options, _, _ = generate_cq(ex_)
    ex_["ap"] = ap
    ex_["cq_options"] = cq_options
    return ex_


def json_dataset_weak_labeling(json_dataset):
    ds = json_to_dataset(json_dataset)
    ds = ds.map(generate_machine_cq, with_indices=True)
    res = []
    for ex in ds:
        orig_idx = ex['orig_idx']
        orig_ex = json_dataset[orig_idx]
        orig_ex["ap"] = ex["ap"]
        orig_ex["cq_options"] = ex["cq_options"]
        res.append(orig_ex)
    return res


def json_to_dataset(json_dataset):
    ambig_qs = []
    orig_idx_in_ambigNQ = []
    for idx, i in enumerate(json_dataset):
        if any(j["type"] != "singleAnswer" for j in i["annotations"]):
            ambig_qs.append(i)
            orig_idx_in_ambigNQ.append(idx)
    ref = defaultdict(list, [])
    for inst in ambig_qs:
        dqs = []
        answers = []
        num_of_sets_of_multipleQAs = 0
        for annot in inst["annotations"]:
            if annot["type"] == "multipleQAs":
                num_of_sets_of_multipleQAs += 1
                for qapair in annot["qaPairs"]:
                    # For HTML compatability, remove "
                    dqs.append(qapair["question"].replace("\"", ""))
                    answers.append([i.replace("\"", "") for i in qapair["answer"]])


        assert len(dqs) == len(answers)
        # remove redundant dq-answer pairs based on answers
        if num_of_sets_of_multipleQAs > 1:
            dqs, answers = remove_redundant_pairs(dqs, answers)
        assert len(dqs) == len(answers)
        gold_passages = extract_gold_passage(inst['viewed_doc_titles'], inst['used_queries'])

        ref['aq'].append(inst["question"])
        ref['dqs'].append(dqs)
        ref['answers'].append(answers)
        ref['viewed_doc_titles'].append(inst['viewed_doc_titles'])
        ref['gold_passages'].append(gold_passages)
    ref['orig_idx'] = orig_idx_in_ambigNQ  # w.r.t. ambigNQ. NOT NQ dataset.
    dataset = Dataset.from_dict(ref)
    return dataset


def augment_json_with_cq(json_dataset, hf_dataset):
    for ex in hf_dataset:
        json_dataset[ex['orig_idx']]["clarification_question"] = ex["machine_cq"]
        json_dataset[ex['orig_idx']]["clarification_answers"] = ex["answers"]
    return json_dataset


def extract_gold_passage(viewed_doc_titles, used_queries):
    results_all = {}
    for i in used_queries:
        for result in i['results']:
            title, snippet = result['title'], result['snippet']
            results_all[title] = snippet
    candidate_titles = results_all.keys()
    gold_passages = []
    for viewed_doc_title in viewed_doc_titles:
        title = find_the_most_similar_title(viewed_doc_title, candidate_titles)
        passage = results_all[title]
        gold_passages.append((title, passage))
    return gold_passages



def get_leading_wh(doc):
    leading_word = doc[0].text.lower()
    wh_words = ["When", "Where", "Who", "Which", "What"]
    wh_words = {i.lower(): i for i in wh_words}
    return wh_words.get(leading_word, None)


def normalize_questions(example, propns):
    example['aq_'] = substitute_propns(normalize_text(example['aq']), propns)
    example['dqs_'] = []
    for dq in example['dqs']:
        dq_ = substitute_propns(normalize_text(dq), propns)
        example['dqs_'].append(dq_)
    return example


def normalize_text(text):
    text = text.lower().replace("?", " ?")
    text = re.sub("\"", " \" ", text)
    text = re.sub("\+", " + ", text)
    text = re.sub(",", " , ", text)
    text = re.sub(":", " : ", text)
    text = re.sub("/", " / ", text)
    text = re.sub("\'s", " \'s ", text)
    text = re.sub('\s+', ' ', text)
    return text


def get_roots_of_candidates(candidates, propn_dict):
    roots_of_candidates = {}
    for cand in candidates:
        root = 0
        for propn, placeholder in propn_dict.items():
            if is_textual_subspan(propn, cand):
                root = placeholder + "\t" + propn
                break
        if root == 0:
            root = propn_dict.get(cand, get_root(NLP(cand)))
        roots_of_candidates[root] = cand
    return roots_of_candidates


def add_candidates(example):
    spans_of_proper_nouns = []
    aq = normalize_text(example['aq'])
    aq_doc = NLP(aq)
    aq_tokens = get_tokens(aq_doc)

    for title in example['viewed_doc_titles']:
        title_doc = NLP(normalize_text(title))
        title_tokens = get_tokens(title_doc)
        _, ids = lcs(title_tokens, aq_tokens)
        spans = merge_meaningful_adjacency(ids, aq_doc)
        for span in spans:
            spans_of_proper_nouns.append(span)
    spans_of_nouns_by_titles = cleanse_list_by_merging(spans_of_proper_nouns)
    example["nouns_by_titles"] = [span_to_chunk(aq_doc, span) for span in spans_of_nouns_by_titles]
    aq_ = substitute_propns(aq, example['nouns_by_titles'])
    aq__doc = NLP(aq_.lower())

    # store which idx indicates which propn
    idx_to_propns = {}
    for idx, tok in enumerate(aq__doc):
        propn_token = re.findall("propn[0-9]+", tok.text)
        if len(propn_token) > 0:
            propn_idx = int(propn_token[0][5:])
            idx_to_propns[idx] = example['nouns_by_titles'][propn_idx]

    spans_of_nouns = get_spans_of_noun_chunks(aq__doc)
    if get_leading_wh(aq__doc) is not None:
        spans_of_candidates = cleanse_list_by_merging([[0]] + spans_of_nouns
                                                      + [[get_root(aq__doc, return_idx=True, allow_aux_root=False)]])
    else:
        spans_of_candidates = cleanse_list_by_merging(spans_of_nouns + [[get_root(aq__doc, return_idx=True,
                                                                                  allow_aux_root=False)]])

    spans_of_candidates = sorted(spans_of_candidates, key=lambda x: x[0])


    aq_underlined = ""
    span_heads = [i[0] for i in spans_of_candidates]
    span_tails = [i[-1] for i in spans_of_candidates]
    for idx, tok in enumerate(aq__doc):
        postfix = " " * (len(tok.text_with_ws) - len(tok.text))
        if idx in idx_to_propns:
            tok_ = idx_to_propns[idx]
        else:
            tok_ = tok.text

        if idx in span_heads:
            tok_ = "<ins>" + tok_
        if idx in span_tails:
            tok_ += "</ins>"
        aq_underlined += tok_ + postfix

    specials = {}
    for span in spans_of_candidates:
        for idx in idx_to_propns:
            if idx in span:
                specials[idx] = idx_to_propns[idx] + " "
                idx_to_propns[idx] = "-1"
    candidates = [span_to_chunk(aq__doc, span, specials) for span in spans_of_candidates]
    for idx, propn in idx_to_propns.items():
        if propn != "-1":
            candidates += [propn]

    example['candidates'] = candidates
    example['aq_underlined'] = aq_underlined
    return example


# def add_candidates(example):
#     spans_of_proper_nouns = []
#     aq = normalize_text(example['aq'])
#     aq_doc = NLP(aq)
#     aq_tokens = get_tokens(aq_doc)
#
#     for title in example['viewed_doc_titles']:
#         title_doc = NLP(normalize_text(title))
#         title_tokens = get_tokens(title_doc)
#         _, ids = lcs(title_tokens, aq_tokens)
#         spans = merge_meaningful_adjacency(ids, aq_doc)
#         for span in spans:
#             spans_of_proper_nouns.append(span)
#     spans_of_nouns_by_titles = cleanse_list_by_merging(spans_of_proper_nouns)
#     example["nouns_by_titles"] = [span_to_chunk(aq_doc, span) for span in spans_of_nouns_by_titles]
#     aq_ = substitute_propns(aq, example['nouns_by_titles'])
#     aq__doc = NLP(aq_.lower())
#
#     # store which idx indicates which propn
#     idx_to_propns = {}
#     for idx, tok in enumerate(aq__doc):
#         propn_token = re.findall("propn[0-9]+", tok.text)
#         if len(propn_token) > 0:
#             propn_idx = int(propn_token[0][5:])
#             idx_to_propns[idx] = example['nouns_by_titles'][propn_idx]
#
#     spans_of_nouns = get_spans_of_noun_chunks(aq__doc)
#     spans_of_candidates = cleanse_list_by_merging(spans_of_nouns
#                                                   + [[get_root(aq__doc, return_idx=True)]])
#     specials = {}
#     for span in spans_of_candidates:
#         for idx in idx_to_propns:
#             if idx in span:
#                 specials[idx] = idx_to_propns[idx] + " "
#                 idx_to_propns[idx] = "-1"
#     candidates = [span_to_chunk(aq__doc, span, specials) for span in spans_of_candidates]
#     for idx, propn in idx_to_propns.items():
#         if propn != "-1":
#             candidates += [propn]
#
#     wh_word = get_leading_wh(aq__doc)
#     if wh_word is not None and wh_word.lower() not in candidates:
#         candidates.append(wh_word)
#
#     example['candidates'] = candidates
#     return example



#########
def generate_cq(ex):
    ex_ = normalize_questions(ex, ex["nouns_by_titles"])
    aq_ = ex_['aq_']
    candidates = ex_['candidates']

    # print(aq_)
    print(red("AQ: "), ex_['aq'])
    aq_doc = NLP(aq_.lower())
    aq_wh_word = get_leading_wh(aq_doc)
    aq_tokens = get_tokens(aq_doc)

    propn_dict = {propn: f"propn{idx}" for idx, propn in enumerate(ex_['nouns_by_titles'])}

    roots_of_cands = get_roots_of_candidates(candidates, propn_dict)
    temp = {k: highlight_part(v, k, True) for k, v in roots_of_cands.items()}
    print(red("AP candidates: "), " | ".join([v for k, v in temp.items()]))

    scores = defaultdict(list, [])

    print(red("WIKI TITLEs: "), ex_['viewed_doc_titles'])
    print(red("Detected PROPNs: "), ex_['nouns_by_titles'])
    print("-----")

    cq_options = []
    dq_wh_words = []
    assert len(ex["dqs_"]) == len(ex_["answers"])
    for idx_dq, dq_ in enumerate(ex_['dqs_']):

        print(red("DQ: "), dq_)
        dq_doc = NLP(dq_.lower())
        # For wh-exception handling
        dq_wh_word = get_leading_wh(dq_doc)
        dq_wh_words.append(dq_wh_word)
        dq_tokens = get_tokens(dq_doc)
        ids_aq, ids_dq = lcs(aq_tokens, dq_tokens)

        ids_aq_only = [i for i in range(len(aq_doc)) if i not in ids_aq]
        ids_dq_only = [i for i in range(len(dq_doc)) if i not in ids_dq]
        dp_spans_aq = merge_meaningful_adjacency(ids_aq_only, aq_doc)
        dp_spans_dq = merge_meaningful_adjacency(ids_dq_only, dq_doc)

        # For the substitution case
        span_pairs = get_span_pairs(ids_aq, dp_spans_aq, ids_dq, dp_spans_dq)

        dps = []  # list of different parts.
        for dp_span_aq, dp_span_dq in span_pairs:
            dp_dq = scoring(dp_span_aq, aq_doc, dp_span_dq, dq_doc, idx_dq, roots_of_cands, scores)
            dps.append(substitute_propns(dp_dq, ex_["nouns_by_titles"], reverse=True))
        if len(dps) > 0:
            cq_options.append(" ".join(dps))
            # print(red("edited parts: "), dps)
            # print()
        else:
            cq_options.append(dq_)

    root = get_root(aq_doc)
    ap, final_score, reasons, contribution = summarize_score(scores, roots_of_cands, cq_options,
                                                             aq_root=root,
                                                             aq_wh_word=aq_wh_word)

    print(red("score: "), " | ".join([f"{v}: {final_score[k]}" for k, v in temp.items()]))
    cq = write_cq_in_NL(ap, cq_options)
    print(red("CQ: "), cq)

    assert len(cq_options) == len(ex_["answers"])
    return ap, cq_options, reasons, contribution


def check_against_gold_exs(gold_examples, ds, idx=-1):
    def test(idx):
        if idx not in gold_examples:
            return
        ex = ds[idx]
        with HiddenPrints():
            ex_ = add_candidates(ex)
            ap, options, _, _ = generate_cq(ex_)
            pred = f"{ap}\t"+"\t".join(options)
        if pred != gold_examples[idx]:
            print(idx)
            print(pred)
            print(gold_examples[idx])
    if idx > 0:
        test(idx)
    else:
        for idx in gold_examples:
            test(idx)


def mark_aq_with_ap(aq_underlined, ap):
    if not ap:
        return aq_underlined
    ap_ = ap.lower()
    aq_marked = re.sub(f"<ins>{ap_}</ins>", f"<strong><ins>{ap}</ins></strong>", aq_underlined)
    return aq_marked


def concat_passages(passages):
    return "<hr>".join(["<br>".join([f"[{passage[0]}]"] + passage[1:]) for passage in passages])


def rewrite_qaPairs_by_append_at_end(ap, options, I_s, aq, answer_set):
    new_qaPairs = []
    for m, n_ in I_s.items():
        n, score = n_
        if ap == "SPECIFY":
            question = f"{aq} Specifically, {options[m]}'."
        else:
            question = f"{aq} Here, '{ap}' means '{options[m]}'."
        answer = answer_set[n]
        new_qaPairs.append({"question": question, "answer": answer})
    return new_qaPairs


def update_qaPairs_by_middle_insertion(ap, options, I_s, aq, answer_set):
    def replace_proxy(aq, ap, cp):
        if ap == "SPECIFY":
            return aq + " " + cp
        if len(re.findall(ap, aq, re.IGNORECASE)) > 0:
            return re.sub(ap, cp, aq, flags=re.IGNORECASE)
        else:
            _, aq_ = lcs(ap.lower(), aq.lower())
            return aq[:aq_[0]] + cp + aq[aq_[-1]:]
    new_qaPairs = []
    for m, n_ in I_s.items():
        n, score = n_
        question = replace_proxy(aq, ap, ap + f" {options[m]}")
        answer = answer_set[n]
        new_qaPairs.append({"question": question, "answer": answer})
    return new_qaPairs


def update_dqs_by_cq(AmbigNQ_data_path, predicted_cq_path, output_dir, revision=update_qaPairs_by_middle_insertion):
    AmbigNQ_data_path = Path(AmbigNQ_data_path)
    predicted_cq_path = Path(predicted_cq_path)
    output_dir = Path(output_dir)
    assert "dev" in AmbigNQ_data_path.name
    assert "dev" in predicted_cq_path.name
    
    # AmbigQA_dir = Path("/media/disk1/ksk5693/openQA/AmbigQA/")
    # AmbigNQ_data_path = AmbigQA_dir / "data" / "ambigqa" / "dev_cq.json"
    # predicted_cq_path = AmbigQA_dir / "out" / "deprecated" / "ambignq-cq-gen" / "dev_predictions.json"
    # output_dir = AmbigQA_dir / "data" / "ambigqa"

    dev = json.load(open(AmbigNQ_data_path, encoding="utf-8"))
    cqs = json.load(open(predicted_cq_path, encoding="utf-8"))
    
    # Update dqs
    offset = 0
    data_to_tokenize = {}
    buffer = {}
    ref_opt_lens, gen_opt_lens, indices = [], [], []
    answers = {}
    qg_tokenizer = PTBTokenizer()
    for i, ex in enumerate(dev):
        if all([i['type'] != 'singleAnswer' for i in ex['annotations']]):
            # if len(ex['annotations']) > 1:
            #     annot_idx = max([(idx, annot['qaPairs']) for idx, annot
            #                      in enumerate(ex['annotations'])], key=lambda x: len(x[1]))[0]
            # else:
            #     annot_idx = 0
            answers[i] = ex['clarification_answers']

            ref_cq = ex['clarification_question']
            ref_ap, ref_options = QGData.parse_clarification_question(ref_cq)

            cq = cqs[offset]
            gen_ap, gen_options = QGData.parse_clarification_question(cq)
            data_to_tokenize[f"ref_ap.{i}"] = [{"caption": ref_ap}]
            data_to_tokenize[f"gen_ap.{i}"] = [{"caption": gen_ap}]
            buffer[f"gen_ap.{i}"] = gen_ap
            for j, ref_opt in enumerate(ref_options):
                data_to_tokenize[f"ref_options.{i}.{j}"] = [{"caption": ref_opt}]
            ref_opt_lens.append(len(ref_options))
            for j, gen_opt in enumerate(gen_options):
                data_to_tokenize[f"gen_options.{i}.{j}"] = [{"caption": gen_opt}]
                buffer[f"gen_options.{i}.{j}"] = gen_opt
            gen_opt_lens.append(len(gen_options))

            indices.append(i)
            offset += 1
    all_tokens = qg_tokenizer.tokenize(data_to_tokenize)
    def _get(key):
        return {'sent': [QGData.normalize_answer(value) for value in all_tokens[key]]}

    def _get_match(refs, gens, metric):
        sim_fn = QGData.get_sim_fn(metric)
        I_s = {}
        for m, gen in enumerate(gens):
            # bias toward ordered disambiguation.
            I_s[m] = max([(n, sim_fn(gen, ref) + 0.00001) if m == n else (n, sim_fn(gen, ref))
                          for n, ref in enumerate(refs)], key=lambda x: x[1])
        return I_s

    for idx, info in enumerate(zip(ref_opt_lens, gen_opt_lens, indices)):
        ref_opt_len, gen_opt_len, i = info
        I_s = _get_match([_get(f"ref_options.{i}.{j}") for j in range(ref_opt_len)],
                         [_get(f"gen_options.{i}.{j}") for j in range(gen_opt_len)],
                         metric="N-EDIT")
        ap = buffer[f"gen_ap.{i}"]
        options = [buffer[f"gen_options.{i}.{j}"] for j in range(gen_opt_len)]
        new_qaPairs = revision(ap, options, I_s, dev[i]['question'], answers[i])
        dev[i]['annotations'] = [{"type": 'multipleQAs', 'qaPairs': new_qaPairs}]

    name = AmbigNQ_data_path.name.split(".json")[0]
    outfile_name = f"{name}_with_dqs_by_{predicted_cq_path.parent.name}-{predicted_cq_path.name}"
    out_path = output_dir / outfile_name
    # out_path = path_ambigqa / "data" / "ambigqa" / "dev_cq_with_dqs_by_predicted_cqs.json"

    with open(out_path, 'w') as f:
        json.dump(dev, f)
        print(f"Saved {out_path} \n \t Update dqs in {AmbigNQ_data_path} by cqs in {predicted_cq_path}.")


def update_dqs_by_auto_converted_cq(CAmbigNQ_data_path, output_dir, revision=update_qaPairs_by_middle_insertion):
    CAmbigNQ_data_path = Path(CAmbigNQ_data_path)
    output_dir = Path(output_dir)

    # AmbigQA_dir = Path("/media/disk1/ksk5693/openQA/AmbigQA/")
    # AmbigNQ_data_path = AmbigQA_dir / "data" / "ambigqa" / "dev_cq.json"
    # predicted_cq_path = AmbigQA_dir / "out" / "deprecated" / "ambignq-cq-gen" / "dev_predictions.json"
    # output_dir = AmbigQA_dir / "data" / "ambigqa"
    dev = json.load(open(CAmbigNQ_data_path, encoding="utf-8"))
    answers = {}
    for i, ex in enumerate(dev):
        if all([i['type'] != 'singleAnswer' for i in ex['annotations']]):
            cq_ans = ex['clarification_answers']
            answers[i] = cq_ans
            ref_cq = ex['clarification_question']
            ref_ap, ref_options = QGData.parse_clarification_question(ref_cq)
            assert len(ref_options) == len(cq_ans)
            I_s = dict((i, (i, 0)) for i in range(len(ref_options)))
            new_qaPairs = revision(ref_ap, ref_options, I_s, dev[i]['question'], answers[i])
            dev[i]['annotations'] = [{"type": 'multipleQAs', 'qaPairs': new_qaPairs}]

    name = CAmbigNQ_data_path.name.split(".json")[0]
    outfile_name = f"{name}_with_dqs_by_autoconversion.json"
    out_path = output_dir / outfile_name
    # out_path = path_ambigqa / "data" / "ambigqa" / "dev_cq_with_dqs_by_predicted_cqs.json"

    with open(out_path, 'w') as f:
        json.dump(dev, f)
        print(f"Saved {out_path} \n \t Update dqs in {CAmbigNQ_data_path} by autoconverted cqs.")


# If a user knows its desired goal but he didn't ask properly.
# Assuming
def update_dqs_by_aq(CAmbigNQ_data_path, output_dir):
    CAmbigNQ_data_path = Path(CAmbigNQ_data_path)
    output_dir = Path(output_dir)

    # AmbigQA_dir = Path("/media/disk1/ksk5693/openQA/AmbigQA/")
    # AmbigNQ_data_path = AmbigQA_dir / "data" / "ambigqa" / "dev_cq.json"
    # predicted_cq_path = AmbigQA_dir / "out" / "deprecated" / "ambignq-cq-gen" / "dev_predictions.json"
    # output_dir = AmbigQA_dir / "data" / "ambigqa"
    dev = json.load(open(CAmbigNQ_data_path, encoding="utf-8"))
    answers = {}
    for i, ex in enumerate(dev):
        if all([i['type'] != 'singleAnswer' for i in ex['annotations']]):
            cq_ans = ex['clarification_answers']
            answers[i] = cq_ans
            ref_cq = ex['clarification_question']
            ref_ap, ref_options = QGData.parse_clarification_question(ref_cq)
            assert len(ref_options) == len(cq_ans)
            new_qaPairs = [{"question": ex["question"], "answer": ex["nq_answer"]}]
            dev[i]['annotations'] = [{"type": 'multipleQAs', 'qaPairs': new_qaPairs}]

    name = CAmbigNQ_data_path.name.split(".json")[0]
    outfile_name = f"{name}_with_dqs_by_aqs.json"
    out_path = output_dir / outfile_name

    with open(out_path, 'w') as f:
        json.dump(dev, f)
        print(f"Saved {out_path} \n \t Update dqs in {CAmbigNQ_data_path} by aqs.")


# MA + DQs


# DATA PREPROCESSING FUNCTIONS WITH EXAMPLE INPUT.
# def add_candidates(example):
#     proper_nouns = []
#     aq = example['aq'].replace("?", " ?")
#     aq_doc = NLP(aq.lower())
#     aq_tokens = get_tokens(aq_doc)
#     for title in example['viewed_doc_titles']:
#         title_doc = NLP(title.lower())
#         title_tokens = get_tokens(title_doc)
#         ids = lcs(title_tokens, aq_tokens)
#         spaned_ids = merge_adjacency(ids)
#         for span in spaned_ids:
#             if not is_meaningful_span(aq_doc, span):
#                 continue
#             noun = ''.join([token.text_with_ws for idx, token in enumerate(aq_doc) if idx in span])
#             proper_nouns.append(noun.strip())
#     example["nouns_by_titles"] = cleanse_list_by_merging(proper_nouns)
#     aq_ = substitute_propns(aq, example['nouns_by_titles'])
#     aq__doc = NLP(aq_.lower())
#     example['nouns'] = []
#     print(get_noun_chunks(aq__doc))
#     for noun in get_noun_chunks(aq__doc):
#         for propn in re.findall("propn[0-9]+", noun):
#             idx = int(propn[5:])
#             noun = re.sub(propn, example["nouns_by_titles"][idx], noun)
#         example['nouns'].append(noun)
#     example['candidates'] = cleanse_list_by_merging(example['nouns'] + example["nouns_by_titles"] + [get_root(aq__doc)])
#     return example





# def extract_proper_nouns(doc):
#     pos = [tok.i for tok in doc if tok.pos_ == "PROPN"]
#     consecutives = []
#     current = []
#     for elt in pos:
#         if len(current) == 0:
#             current.append(elt)
#         else:
#             if current[-1] == elt - 1:
#                 current.append(elt)
#             else:
#                 consecutives.append(current)
#                 current = [elt]
#     if len(current) != 0:
#         consecutives.append(current)
#     return [doc[consecutive[0]:consecutive[-1] + 1] for consecutive in consecutives]


# def extract_proper_nouns(doc):
#     pos = [tok.i for tok in doc if tok.pos_ == "PROPN"]
#     consecutives = []
#     current = []
#     for elt in pos:
#         if len(current) == 0:
#             current.append(elt)
#         else:
#             if current[-1] == elt - 1:
#                 current.append(elt)
#             else:
#                 consecutives.append(current)
#                 current = [elt]
#     if len(current) != 0:
#         consecutives.append(current)
#     return [doc[consecutive[0]:consecutive[-1] + 1] for consecutive in consecutives]

# def leading_wh_changed(aq_wh_word, dq_wh_words):
#     return any(aq_wh_word != dq_wh_word for dq_wh_word in dq_wh_words)

