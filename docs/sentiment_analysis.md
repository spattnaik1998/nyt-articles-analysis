# Sentiment Analysis - Multi-Model Interface

## Overview

The sentiment analysis module (`src/models/sentiment.py`) provides a unified interface for classifying sentiment using multiple transformer models:

- **FinBERT**: Financial sentiment (ProsusAI/finbert)
- **FinBERT-Tone**: Financial tone analysis (yiyanghkust/finbert-tone)
- **DistilRoBERTa**: General financial news sentiment
- **PoliBERT**: Political bias analysis

All implementations follow patterns from the NYT analysis notebook with GPU auto-detection.

## Quick Start

### 1. Single Text Classification

```python
from src.models.sentiment import classify_sentiment

# Classify with FinBERT
result = classify_sentiment(
    "The stock market rallied today on strong earnings",
    model_key='finbert'
)

print(f"Label: {result['label']}")  # positive
print(f"Score: {result['score']:.3f}")  # 0.954
```

### 2. Batch Classification

```python
from src.models.sentiment import classify_sentiment

texts = [
    "Market crashes amid recession fears",
    "Economy shows steady growth",
    "Financial markets remain volatile"
]

results = classify_sentiment(texts, model_key='finbert')

for text, result in zip(texts, results):
    print(f"{text[:30]}... -> {result['label']} ({result['score']:.3f})")
```

### 3. Multi-Model Analysis

```python
import pandas as pd
from src.models.sentiment import batch_infer

df = pd.DataFrame({
    'text': [
        "Strong economic indicators boost market confidence",
        "Trade tensions escalate causing market uncertainty"
    ]
})

# Run multiple models
result_df = batch_infer(
    df,
    text_col='text',
    models=['finbert', 'distilroberta', 'finbert_tone']
)

print(result_df[['text', 'finbert_label', 'distilroberta_label']])
```

### 4. Generate Comparison Report

```python
from src.models.sentiment import model_comparison_report

# After running batch_infer
report = model_comparison_report(
    result_df,
    models=['finbert', 'distilroberta'],
    output_path='data/sentiment_report.txt'
)

print(f"Agreement rate: {report['agreement_analysis']['perfect_agreement_rate']:.1%}")
```

### 5. Run Example Script

```bash
# Basic usage with sample data
python examples/sentiment_report.py

# Analyze specific dataset
python examples/sentiment_report.py --input data/preprocessed.parquet --sample 100

# Compare all models
python examples/sentiment_report.py --models finbert distilroberta finbert_tone
```

## Available Models

### Model Registry

```python
from src.models.sentiment import list_available_models, get_model_info

# List all models
models = list_available_models()
print(models)
# ['finbert', 'finbert_tone', 'distilroberta', 'polibert']

# Get model info
info = get_model_info('finbert')
print(info['description'])  # Financial sentiment analysis
print(info['labels'])  # ['positive', 'negative', 'neutral']
```

### Model Details

#### FinBERT
- **HuggingFace**: ProsusAI/finbert
- **Description**: Financial sentiment analysis
- **Labels**: positive, negative, neutral
- **Use Case**: Financial news, earnings reports, market analysis

#### FinBERT-Tone
- **HuggingFace**: yiyanghkust/finbert-tone
- **Description**: Financial tone analysis
- **Labels**: positive, negative, neutral
- **Use Case**: Financial documents, analyst reports

#### DistilRoBERTa
- **HuggingFace**: mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis
- **Description**: General financial news sentiment
- **Labels**: positive, negative, neutral
- **Use Case**: General financial news articles

#### PoliBERT
- **HuggingFace**: bucketresearch/politicalBiasBERT
- **Description**: Political bias analysis
- **Labels**: left, center, right
- **Use Case**: Political news, opinion articles

## Core Functions

### `classify_sentiment(...)`

Classify sentiment for single text or batch of texts.

**Parameters:**
- `texts` (str or List[str]): Text(s) to classify
- `model_key` (str): Model to use ('finbert', 'finbert_tone', 'distilroberta', 'polibert')
- `batch_size` (int): Batch size for processing (default: 32)
- `max_length` (int): Maximum sequence length (default: 512)
- `device` (str, optional): Device to use ('cuda', 'mps', 'cpu'). Auto-detect if None.
- `return_scores` (bool): Return confidence scores (default: True)

**Returns:**
- Dict or List[Dict]: Results with 'label' and 'score'

**Example:**
```python
# Single text
result = classify_sentiment(
    "Market volatility increases",
    model_key='finbert'
)
print(result)
# {'label': 'negative', 'score': 0.823}

# Batch
results = classify_sentiment(
    ["Market up", "Market down"],
    model_key='finbert',
    batch_size=16
)
print(results)
# [{'label': 'positive', 'score': 0.89}, {'label': 'negative', 'score': 0.92}]

# Without scores
result = classify_sentiment("Test", model_key='finbert', return_scores=False)
print(result)
# {'label': 'neutral'}
```

### `batch_infer(...)`

Run batch sentiment inference with multiple models on a DataFrame.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `text_col` (str): Column containing text (default: 'combined_text')
- `models` (List[str]): List of model keys (default: ['finbert', 'distilroberta', 'finbert_tone'])
- `batch_size` (int): Batch size (default: 32)
- `max_length` (int): Max sequence length (default: 512)
- `device` (str, optional): Device to use. Auto-detect if None.
- `verbose` (bool): Show progress bars (default: True)

**Returns:**
- pd.DataFrame: DataFrame with added columns: {model}_label, {model}_score

**Example:**
```python
import pandas as pd
from src.models.sentiment import batch_infer

df = pd.DataFrame({
    'article_text': [
        "Strong earnings boost market",
        "Recession fears grow",
        "Stable economic outlook"
    ]
})

# Run FinBERT and DistilRoBERTa
result_df = batch_infer(
    df,
    text_col='article_text',
    models=['finbert', 'distilroberta'],
    batch_size=32,
    verbose=True
)

print(result_df.columns)
# ['article_text', 'finbert_label', 'finbert_score',
#  'distilroberta_label', 'distilroberta_score']

# Analyze results
print(result_df['finbert_label'].value_counts())
# positive    2
# negative    1
```

### `model_comparison_report(...)`

Generate comprehensive comparison report for multiple models.

**Parameters:**
- `df` (pd.DataFrame): DataFrame with sentiment results (from batch_infer)
- `models` (List[str], optional): Models to compare. Auto-detect if None.
- `output_path` (str, optional): Path to save report text file
- `min_score_threshold` (float): Min score for confident predictions (default: 0.5)
- `verbose` (bool): Print report to console (default: True)

**Returns:**
- Dict: Report statistics with keys:
  - `total_documents`: Total documents analyzed
  - `models_compared`: List of models
  - `label_distributions`: Per-model label counts
  - `score_stats`: Per-model score statistics
  - `agreement_analysis`: Agreement/disagreement metrics
  - `disagreements`: List of disagreement examples

**Example:**
```python
from src.models.sentiment import batch_infer, model_comparison_report

# Run inference
df = batch_infer(df, models=['finbert', 'distilroberta'])

# Generate report
report = model_comparison_report(
    df,
    models=['finbert', 'distilroberta'],
    output_path='data/comparison_report.txt',
    verbose=True
)

# Access statistics
print(f"Total: {report['total_documents']}")
print(f"Agreement: {report['agreement_analysis']['perfect_agreement_rate']:.1%}")

# Check disagreements
disagreements = report['disagreements']
for dis in disagreements[:5]:
    print(f"Row {dis['index']}:")
    print(f"  FinBERT: {dis['finbert_label']} ({dis['finbert_score']:.3f})")
    print(f"  DistilRoBERTa: {dis['distilroberta_label']} ({dis['distilroberta_score']:.3f})")
```

## GPU Support

### Automatic Device Detection

The module automatically detects and uses available hardware:

```python
from src.models.sentiment import get_device

device = get_device()
print(f"Using: {device}")
# CUDA GPU: "cuda"
# Apple MPS: "mps"
# CPU: "cpu"
```

### Force Specific Device

```python
# Force CPU
result = classify_sentiment("text", model_key='finbert', device='cpu')

# Force CUDA
result = classify_sentiment("text", model_key='finbert', device='cuda')
```

### Performance Comparison

| Device | 1000 Texts | 10000 Texts |
|--------|-----------|-------------|
| CPU | ~2 min | ~20 min |
| CUDA GPU | ~15 sec | ~2 min |
| Apple MPS | ~30 sec | ~5 min |

## Complete Workflow Example

```python
import pandas as pd
from src.ingest.load_nyt import load_nyt_csv
from src.preprocess.text import preprocess_dataframe
from src.models.sentiment import batch_infer, model_comparison_report

# 1. Load and preprocess data
df = load_nyt_csv('data/nyt-metadata.csv')
df = preprocess_dataframe(df)

# 2. Filter to business articles
business_df = df[df['section_name'] == 'Business'].copy()

# Sample for testing
sample_df = business_df.sample(n=100, random_state=42)

# 3. Run sentiment analysis with multiple models
result_df = batch_infer(
    sample_df,
    text_col='combined_text',
    models=['finbert', 'distilroberta', 'finbert_tone'],
    batch_size=32,
    verbose=True
)

# 4. Save results
result_df.to_csv('data/business_sentiment.csv', index=False)

# 5. Generate comparison report
report = model_comparison_report(
    result_df,
    models=['finbert', 'distilroberta', 'finbert_tone'],
    output_path='data/business_sentiment_report.txt',
    verbose=True
)

# 6. Analyze results
print(f"\nLabel Distributions:")
for model in ['finbert', 'distilroberta']:
    print(f"\n{model}:")
    print(result_df[f'{model}_label'].value_counts())

print(f"\nAgreement Analysis:")
print(f"Perfect Agreement: {report['agreement_analysis']['perfect_agreement_rate']:.1%}")
print(f"Disagreements: {report['agreement_analysis']['disagreement_count']}")

# 7. Find high-confidence predictions
high_conf = result_df[result_df['finbert_score'] > 0.9]
print(f"\nHigh Confidence Predictions: {len(high_conf)}")

# 8. Analyze disagreements
disagreements_df = result_df[
    result_df['finbert_label'] != result_df['distilroberta_label']
]
print(f"Disagreement Cases: {len(disagreements_df)}")
```

## Example Script Usage

### Basic Usage

```bash
# Run with sample data (10 articles)
python examples/sentiment_report.py
```

### Custom Input

```bash
# Analyze preprocessed data (sample 100)
python examples/sentiment_report.py \
    --input data/preprocessed.parquet \
    --sample 100

# Specific section
python examples/sentiment_report.py \
    --input data/preprocessed.parquet \
    --section Business \
    --sample 50
```

### Model Selection

```bash
# Use specific models
python examples/sentiment_report.py \
    --models finbert distilroberta

# All models
python examples/sentiment_report.py \
    --models finbert finbert_tone distilroberta polibert
```

### Save Outputs

```bash
# Custom output paths
python examples/sentiment_report.py \
    --input data/preprocessed.parquet \
    --output data/my_results.csv \
    --report data/my_report.txt \
    --sample 200
```

### Advanced Options

```bash
# Custom batch size and text column
python examples/sentiment_report.py \
    --input data/preprocessed.parquet \
    --text-col cleaned_text \
    --batch-size 16 \
    --sample 500

# Skip report generation
python examples/sentiment_report.py \
    --input data/preprocessed.parquet \
    --no-report
```

## Understanding the Comparison Report

### Report Structure

```
======================================================================
SENTIMENT MODEL COMPARISON REPORT
======================================================================

Total Documents: 100
Models Compared: finbert, distilroberta

----------------------------------------------------------------------
LABEL DISTRIBUTIONS
----------------------------------------------------------------------

FINBERT (Financial sentiment analysis):
  positive       : 45 (45.0%)
  neutral        : 32 (32.0%)
  negative       : 23 (23.0%)
  Score stats: mean=0.856, median=0.891

DISTILROBERTA (General financial news sentiment):
  positive       : 48 (48.0%)
  neutral        : 29 (29.0%)
  negative       : 23 (23.0%)
  Score stats: mean=0.823, median=0.854

----------------------------------------------------------------------
AGREEMENT ANALYSIS
----------------------------------------------------------------------

Perfect Agreement: 87 / 100 (87.0%)
Disagreements: 13 (13.0%)

Pairwise Agreement (finbert vs distilroberta):
  87 / 100 (87.0%)

----------------------------------------------------------------------
TOP DISAGREEMENTS (showing up to 20)
----------------------------------------------------------------------

1. Row 45 (avg_score=0.612):
   finbert        : positive   (score=0.678)
   distilroberta  : neutral    (score=0.546)
   Text: The market showed mixed signals with some sectors...

...
```

### Interpreting Results

**High Agreement (>90%)**
- Models are consistent
- Clear sentiment in texts
- Reliable predictions

**Moderate Agreement (70-90%)**
- Some ambiguity in texts
- Model differences in interpretation
- Review disagreements for insights

**Low Agreement (<70%)**
- Highly ambiguous texts
- Different model focuses
- Consider domain-specific model

## Troubleshooting

### Issue 1: Transformers Not Available

**Error:**
```
ImportError: Transformers required. Install with: pip install transformers torch
```

**Solution:**
```bash
pip install transformers torch
```

### Issue 2: CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
```python
# Reduce batch size
result_df = batch_infer(df, batch_size=8)  # Instead of 32

# Or force CPU
result_df = batch_infer(df, device='cpu')
```

### Issue 3: Model Download Issues

**Problem:** Model fails to download

**Solution:**
```python
# Set HuggingFace cache directory
import os
os.environ['TRANSFORMERS_CACHE'] = '/path/to/cache'

# Or download manually first
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained('ProsusAI/finbert')
tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
```

### Issue 4: Slow Inference

**Problem:** Processing too slow

**Solutions:**
1. Use GPU: Ensure CUDA/MPS available
2. Increase batch size: `batch_size=64`
3. Reduce max_length: `max_length=256`
4. Sample data first: `df.sample(n=1000)`

## Performance Optimization

### Batch Size Tuning

```python
# Small dataset (<100) - Small batch
batch_infer(df, batch_size=8)

# Medium dataset (100-1000) - Medium batch
batch_infer(df, batch_size=32)

# Large dataset (>1000) - Large batch (if GPU available)
batch_infer(df, batch_size=64)
```

### GPU Memory Management

```python
import torch

# Clear cache between models
torch.cuda.empty_cache()

# Monitor memory
if torch.cuda.is_available():
    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

## Integration with Other Modules

### With Preprocessing

```python
from src.preprocess.text import preprocess_dataframe
from src.models.sentiment import batch_infer

# Preprocess first
df = preprocess_dataframe(df)

# Then analyze sentiment
df = batch_infer(df, text_col='cleaned_text')
```

### With Topic Models

```python
from src.models.topic_models import run_lda
from src.models.sentiment import batch_infer

# Run both topic modeling and sentiment
df = batch_infer(df, models=['finbert'])

# Group by sentiment
for sentiment in ['positive', 'negative', 'neutral']:
    subset = df[df['finbert_label'] == sentiment]
    docs = subset['combined_text'].tolist()

    # Topic modeling per sentiment
    lda_model, topics_df, _, _ = run_lda(
        docs,
        num_topics=5,
        output_dir=f'data/topics_{sentiment}'
    )
```

### With Similarity Search

```python
from src.models.similarity import recommend_by_embedding
from src.models.sentiment import batch_infer

# Get sentiment
df = batch_infer(df, models=['finbert'])

# Find similar positive articles
positive_articles = df[df['finbert_label'] == 'positive']
query_text = positive_articles.iloc[0]['combined_text']

similar = recommend_by_embedding(query_text, top_k=10)
```

## Next Steps

After implementing sentiment analysis:

1. **Build temporal sentiment analysis**
   - Track sentiment changes over time
   - Analyze trends by section/topic
   - Visualize sentiment evolution

2. **Create sentiment dashboard**
   - Real-time sentiment monitoring
   - Section-wise sentiment breakdowns
   - Alert on sentiment shifts

3. **Integrate with API**
   - FastAPI endpoints for sentiment classification
   - Batch processing endpoints
   - Model comparison API

4. **Advanced analytics**
   - Sentiment + topic modeling
   - Sentiment + named entities
   - Predictive sentiment modeling

## Summary

**Features Implemented:**
- ✅ Unified interface for 4 sentiment models
- ✅ GPU auto-detection (CUDA/MPS/CPU)
- ✅ Batch inference with multiple models
- ✅ Model comparison report with disagreement analysis
- ✅ Complete example script and documentation

**Models Available:**
- FinBERT (financial sentiment)
- FinBERT-Tone (financial tone)
- DistilRoBERTa (general financial)
- PoliBERT (political bias)

**Ready for:**
- Large-scale sentiment analysis
- Multi-model comparison studies
- Production deployment
- API integration
