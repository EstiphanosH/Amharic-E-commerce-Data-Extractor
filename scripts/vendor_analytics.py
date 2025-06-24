import pandas as pd
from dateutil.relativedelta import relativedelta
import re

def extract_prices(text, nlp_pipeline):
    """Extract prices using NER model"""
    results = nlp_pipeline(text)
    prices = []
    for ent in results:
        if ent["entity_group"] == "PRICE":
            # Extract numeric values from price strings
            numbers = re.findall(r'\d+', ent["word"])
            if numbers:
                prices.append(float(''.join(numbers)))
    return prices

def calculate_vendor_metrics(df, nlp_pipeline):
    """Calculate KPIs for each vendor"""
    metrics = []
    
    for vendor, group in df.groupby('channel'):
        # Activity metrics
        group['date'] = pd.to_datetime(group['date'])
        post_freq = group.resample('W', on='date').size().mean()
        
        # Engagement metrics
        avg_views = group['views'].mean()
        top_post = group.loc[group['views'].idxmax()]
        
        # Business metrics
        all_prices = []
        for text in group['text']:
            all_prices.extend(extract_prices(text, nlp_pipeline))
        avg_price = sum(all_prices) / len(all_prices) if all_prices else 0
        
        metrics.append({
            'vendor': vendor,
            'avg_views': avg_views,
            'post_freq': post_freq,
            'avg_price': avg_price,
            'top_post_views': top_post['views'],
            'top_post_product': extract_products(top_post['text'], nlp_pipeline)[0] if group['text'].any() else ''
        })
    
    return pd.DataFrame(metrics)

def lending_scorecard(metrics_df):
    """Generate lending scorecard with weighted metrics"""
    # Normalize metrics
    metrics_df['views_norm'] = metrics_df['avg_views'] / metrics_df['avg_views'].max()
    metrics_df['freq_norm'] = metrics_df['post_freq'] / metrics_df['post_freq'].max()
    
    # Calculate score (customizable weights)
    metrics_df['lending_score'] = (
        0.5 * metrics_df['views_norm'] + 
        0.3 * metrics_df['freq_norm'] + 
        0.2 * (metrics_df['avg_price'] / metrics_df['avg_price'].max())
    )
    
    return metrics_df.sort_values('lending_score', ascending=False)