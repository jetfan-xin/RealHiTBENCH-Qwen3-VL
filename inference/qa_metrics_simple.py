"""
Simplified QA metrics for Qwen3-VL evaluation.
This version avoids the evaluate library dependency issues.
"""
import re
import string
from collections import Counter


def process_decimal(s):
    """Round decimal numbers to 1 decimal place."""
    pattern = r'\b\d+\.\d+\b'
    
    def round_match(match):
        number = float(match.group())
        rounded_number = round(number, 1)
        return str(rounded_number)

    result = re.sub(pattern, round_match, s)
    return result


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def word_level_f1_score(reference, prediction):
    """Compute word-level F1 score between reference and prediction."""
    prediction_words = prediction.split()
    reference_words = reference.split()
    
    common = Counter(prediction_words) & Counter(reference_words)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = 1.0 * num_same / len(prediction_words) if prediction_words else 0
    recall = 1.0 * num_same / len(reference_words) if reference_words else 0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(reference, prediction):
    """Check if prediction exactly matches reference after normalization."""
    return 1.0 if normalize_answer(reference) == normalize_answer(prediction) else 0.0


def rouge_l_score(reference, prediction):
    """Compute ROUGE-L score (longest common subsequence)."""
    def lcs_length(x, y):
        """Compute length of longest common subsequence."""
        m, n = len(x), len(y)
        table = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    table[i][j] = table[i-1][j-1] + 1
                else:
                    table[i][j] = max(table[i-1][j], table[i][j-1])
        return table[m][n]
    
    ref_words = reference.split()
    pred_words = prediction.split()
    
    if not ref_words or not pred_words:
        return 0.0
    
    lcs_len = lcs_length(ref_words, pred_words)
    
    precision = lcs_len / len(pred_words) if pred_words else 0
    recall = lcs_len / len(ref_words) if ref_words else 0
    
    if precision + recall == 0:
        return 0.0
    
    rouge_l = (2 * precision * recall) / (precision + recall)
    return rouge_l


def simple_bleu_score(reference, prediction, max_n=4):
    """Compute simplified BLEU-like score."""
    ref_words = reference.split()
    pred_words = prediction.split()
    
    if not pred_words:
        return 0.0
    
    # Compute n-gram precisions
    precisions = []
    for n in range(1, min(max_n + 1, len(pred_words) + 1)):
        ref_ngrams = Counter([tuple(ref_words[i:i+n]) for i in range(len(ref_words)-n+1)])
        pred_ngrams = Counter([tuple(pred_words[i:i+n]) for i in range(len(pred_words)-n+1)])
        
        common = sum((ref_ngrams & pred_ngrams).values())
        total = sum(pred_ngrams.values())
        
        if total > 0:
            precisions.append(common / total)
        else:
            precisions.append(0)
    
    if not precisions or all(p == 0 for p in precisions):
        return 0.0
    
    # Geometric mean of precisions
    import math
    log_sum = sum(math.log(p) for p in precisions if p > 0) / len([p for p in precisions if p > 0])
    geo_mean = math.exp(log_sum)
    
    # Brevity penalty
    bp = 1.0
    if len(pred_words) < len(ref_words):
        bp = math.exp(1 - len(ref_words) / len(pred_words))
    
    return bp * geo_mean * 100  # Scale to 0-100


class QAMetricSimple:
    """Simplified QA metric class without external dependencies."""
    
    def __init__(self, count_blank=True):
        self.count_blank = count_blank
        print('Initialized simplified QA metrics (F1, EM, ROUGE-L, BLEU)')
    
    def preprocess(self, text):
        """Preprocess text for evaluation."""
        text = process_decimal(text)
        text = normalize_answer(text)
        if len(text) == 0 and self.count_blank:
            text = '#'
        return text
    
    def compute(self, references, predictions):
        """
        Compute all metrics for a batch of references and predictions.
        
        Args:
            references: List of reference answers
            predictions: List of predicted answers
            
        Returns:
            Dictionary with F1, EM, ROUGE-L, and SacreBLEU scores
        """
        if len(references) != len(predictions):
            raise ValueError("References and predictions must have the same length")
        
        f1_scores = []
        em_scores = []
        rouge_l_scores = []
        bleu_scores = []
        
        for ref, pred in zip(references, predictions):
            # Preprocess
            ref_processed = self.preprocess(ref)
            pred_processed = self.preprocess(pred)
            
            # Compute metrics
            f1_scores.append(word_level_f1_score(ref_processed, pred_processed))
            em_scores.append(exact_match_score(ref_processed, pred_processed))
            rouge_l_scores.append(rouge_l_score(ref_processed, pred_processed))
            bleu_scores.append(simple_bleu_score(ref_processed, pred_processed))
        
        # Average scores
        n = len(references)
        return {
            'F1': round(sum(f1_scores) / n, 4) if n > 0 else 0,
            'EM': round(sum(em_scores) / n, 4) if n > 0 else 0,
            'ROUGE-L': round(sum(rouge_l_scores) / n, 4) if n > 0 else 0,
            'SacreBLEU': round(sum(bleu_scores) / n, 4) if n > 0 else 0,
        }


# Alias for compatibility
QAMetric = QAMetricSimple


if __name__ == '__main__':
    # Test the metrics
    metric = QAMetric()
    
    # Test case 1: Exact match
    refs = ["The answer is 42"]
    preds = ["The answer is 42"]
    result = metric.compute(refs, preds)
    print(f"Test 1 (exact match): {result}")
    
    # Test case 2: Partial match
    refs = ["The quick brown fox jumps over the lazy dog"]
    preds = ["The brown fox jumps over the dog"]
    result = metric.compute(refs, preds)
    print(f"Test 2 (partial match): {result}")
    
    # Test case 3: No match
    refs = ["Hello world"]
    preds = ["Goodbye universe"]
    result = metric.compute(refs, preds)
    print(f"Test 3 (no match): {result}")
