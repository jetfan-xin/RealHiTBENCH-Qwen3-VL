import json
from pathlib import Path
from collections import defaultdict

root = Path('/ltstorage/home/pan/4xin/RealHiTBENCH-Qwen3-VL/result/complied')
files = list(root.rglob('results.json'))
print(f'Found {len(files)} results.json files\n')

def aggregate_metrics(metrics_list):
    agg = {}
    keys = set()
    for m in metrics_list:
        keys.update(m.keys())
    for key in keys:
        if key in {'Pass', 'ECR'}:
            vals = []
            for m in metrics_list:
                v = m.get(key)
                if isinstance(v, bool):
                    vals.append(1 if v else 0)
                elif isinstance(v, str):
                    if v.lower() == 'true':
                        vals.append(1)
                    elif v.lower() == 'false':
                        vals.append(0)
                elif isinstance(v, (int, float)):
                    vals.append(1 if v else 0)
            if vals:
                agg[key] = round(sum(vals) / len(metrics_list) * 100, 2)  # percentage
        else:
            vals = [m.get(key) for m in metrics_list if isinstance(m.get(key), (int, float))]
            if vals:
                agg[key] = round(sum(vals) / len(vals), 2)
    return agg

for fp in files:
    try:
        data = json.loads(fp.read_text())
    except Exception as e:
        print(f'Skip {fp}: {e}')
        continue
    
    results = data.get('results', [])
    if not isinstance(results, list):
        print(f'Skip {fp}: results is not a list')
        continue
    
    # Group by QuestionType
    by_type = defaultdict(list)
    for r in results:
        qtype = r.get('QuestionType', 'Unknown')
        metrics = r.get('Metrics', {}) or {}
        by_type[qtype].append(metrics)
    
    # Calculate aggregate metrics by QuestionType
    aggregate_by_type = {qtype: aggregate_metrics(mlist) for qtype, mlist in by_type.items()}
    
    # Also calculate overall average
    all_metrics = [r.get('Metrics', {}) or {} for r in results]
    aggregate_overall = aggregate_metrics(all_metrics)
    
    # Add to the data
    data['aggregate_metrics'] = {
        'by_question_type': aggregate_by_type,
        'overall': aggregate_overall
    }
    
    # Save back
    fp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    
    # Pretty print
    print(f'✓ {fp.relative_to(root)}: {len(results)} samples')
    print(f'  Overall: {aggregate_overall}')
    for qtype, metrics in aggregate_by_type.items():
        print(f'  {qtype}: {metrics}')

print(f'\nDone! Updated all {len(files)} results.json files.')
