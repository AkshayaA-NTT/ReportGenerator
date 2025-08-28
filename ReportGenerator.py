import json
import pandas as pd
import numpy as np
from datetime import datetime
from textblob import TextBlob
import matplotlib.pyplot as plt

def load_jira_tickets(file_path):
    """Load Jira tickets from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if isinstance(data, list):
        tickets = data
    elif 'issues' in data:
        tickets = data['issues']
    else:
        tickets = [data]
    
    return tickets

def parse_date(date_str):
    """Parse Jira date formats"""
    if not date_str:
        return None
    
    formats = [
        '%Y-%m-%dT%H:%M:%S.%f%z',
        '%Y-%m-%dT%H:%M:%S%z', 
        '%Y-%m-%dT%H:%M:%S.%fZ',
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%d %H:%M:%S'
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None

def process_tickets(tickets):
    """Convert raw tickets to DataFrame"""
    processed = []
    
    for ticket in tickets:
        # Extract basic fields
        key = ticket.get('key', '')
        summary = ticket.get('summary', '')
        description = ticket.get('description', '')
        
        # Handle priority (could be string or object)
        priority = ticket.get('priority', 'Unknown')
        if isinstance(priority, dict):
            priority = priority.get('name', 'Unknown')
        
        # Handle status
        status = ticket.get('status', 'Unknown')
        if isinstance(status, dict):
            status = status.get('name', 'Unknown')
        
        # Parse dates
        created = parse_date(ticket.get('created'))
        resolved = parse_date(ticket.get('resolved'))
        
        # Process comments
        comments = ticket.get('comments', [])
        if isinstance(comments, dict):
            comments = comments.get('comments', [])
        
        # Calculate first response time
        first_response_time = None
        comment_text = ""
        
        if comments and created:
            # Sort comments by date
            sorted_comments = []
            for comment in comments:
                comment_date = parse_date(comment.get('created'))
                if comment_date:
                    sorted_comments.append((comment_date, comment.get('body', '')))
            
            sorted_comments.sort(key=lambda x: x[0])
            
            # First response time in hours
            if sorted_comments:
                first_response_time = (sorted_comments[0][0] - created).total_seconds() / 3600
                comment_text = " ".join([c[1] for c in sorted_comments])
        
        # Combine all text for sentiment analysis
        full_text = f"{summary} {description} {comment_text}"
        
        processed.append({
            'key': key,
            'summary': summary,
            'description': description,
            'priority': priority,
            'status': status,
            'created': created,
            'resolved': resolved,
            'first_response_hours': first_response_time,
            'full_text': full_text
        })
    
    return pd.DataFrame(processed)

def analyze_issues(df):
    """Analyze issue metrics"""
    total_issues = len(df)
    status_counts = df['status'].value_counts()
    priority_counts = df['priority'].value_counts()
    
    print("📊 ISSUE ANALYSIS")
    print("-" * 30)
    print(f"Total Issues: {total_issues}")
    print(f"\nBy Status:")
    for status, count in status_counts.items():
        print(f"  {status}: {count} ({count/total_issues*100:.1f}%)")
    
    print(f"\nBy Priority:")
    for priority, count in priority_counts.items():
        print(f"  {priority}: {count} ({count/total_issues*100:.1f}%)")
    
    return {
        'total': total_issues,
        'by_status': status_counts.to_dict(),
        'by_priority': priority_counts.to_dict()
    }

def analyze_response_times(df):
    """Analyze response time metrics"""
    response_data = df.dropna(subset=['first_response_hours'])
    
    if response_data.empty:
        print("⏱️  RESPONSE TIME ANALYSIS")
        print("-" * 30)
        print("No response time data available")
        return {}
    
    avg_response = response_data['first_response_hours'].mean()
    median_response = response_data['first_response_hours'].median()
    
    print("⏱️  RESPONSE TIME ANALYSIS")
    print("-" * 30)
    print(f"Average Response Time: {avg_response:.2f} hours")
    print(f"Median Response Time: {median_response:.2f} hours")
    print(f"Tickets with Response Data: {len(response_data)}")
    
    # Response time by priority
    print("\nBy Priority:")
    priority_response = response_data.groupby('priority')['first_response_hours'].mean()
    for priority, avg_time in priority_response.items():
        print(f"  {priority}: {avg_time:.2f} hours")
    
    return {
        'average_hours': avg_response,
        'median_hours': median_response,
        'by_priority': priority_response.to_dict()
    }

def analyze_sentiment(df):
    """Perform sentiment analysis"""
    sentiments = []
    polarities = []
    
    print("😊 SENTIMENT ANALYSIS")
    print("-" * 30)
    print("Analyzing sentiment... (this may take a moment)")
    
    for text in df['full_text']:
        if pd.isna(text) or not text.strip():
            sentiment = 'neutral'
            polarity = 0
        else:
            blob = TextBlob(str(text))
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                sentiment = 'positive'
            elif polarity < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
        
        sentiments.append(sentiment)
        polarities.append(polarity)
    
    df['sentiment'] = sentiments
    df['polarity'] = polarities
    
    # Calculate distribution
    sentiment_counts = pd.Series(sentiments).value_counts()
    avg_polarity = np.mean(polarities)
    
    print(f"Average Polarity: {avg_polarity:.3f} (-1 to 1 scale)")
    print(f"\nSentiment Distribution:")
    for sentiment, count in sentiment_counts.items():
        percentage = count / len(df) * 100
        print(f"  {sentiment.title()}: {count} ({percentage:.1f}%)")
    
    return {
        'distribution': sentiment_counts.to_dict(),
        'average_polarity': avg_polarity
    }

def calculate_satisfaction_score(df):
    """Calculate overall satisfaction"""
    if 'sentiment' not in df.columns:
        return {}
    
    total = len(df)
    positive = len(df[df['sentiment'] == 'positive'])
    neutral = len(df[df['sentiment'] == 'neutral'])
    negative = len(df[df['sentiment'] == 'negative'])
    
    # Weighted score: positive=100%, neutral=50%, negative=0%
    satisfaction_score = (positive * 100 + neutral * 50) / total
    
    print("📈 SATISFACTION METRICS")
    print("-" * 30)
    print(f"Overall Satisfaction Score: {satisfaction_score:.1f}%")
    print(f"Positive Sentiment: {positive/total*100:.1f}%")
    print(f"Neutral Sentiment: {neutral/total*100:.1f}%")
    print(f"Negative Sentiment: {negative/total*100:.1f}%")
    
    return {
        'overall_score': satisfaction_score,
        'positive_pct': positive/total*100,
        'neutral_pct': neutral/total*100,
        'negative_pct': negative/total*100
    }

def create_charts(df, metrics):
    """Create visualization charts"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Jira Ticket Analysis Dashboard', fontsize=16)
    
    # 1. Status Distribution
    if 'by_status' in metrics.get('issues', {}):
        status_data = metrics['issues']['by_status']
        axes[0,0].pie(status_data.values(), labels=status_data.keys(), autopct='%1.1f%%')
        axes[0,0].set_title('Issues by Status')
    
    # 2. Priority Distribution
    if 'by_priority' in metrics.get('issues', {}):
        priority_data = metrics['issues']['by_priority']
        axes[0,1].bar(priority_data.keys(), priority_data.values())
        axes[0,1].set_title('Issues by Priority')
        axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Sentiment Distribution
    if 'sentiment' in df.columns:
        sentiment_counts = df['sentiment'].value_counts()
        colors = {'positive': 'green', 'neutral': 'yellow', 'negative': 'red'}
        bar_colors = [colors.get(x, 'blue') for x in sentiment_counts.index]
        axes[1,0].bar(sentiment_counts.index, sentiment_counts.values, color=bar_colors)
        axes[1,0].set_title('Sentiment Distribution')
    
    # 4. Response Time Distribution
    if 'first_response_hours' in df.columns:
        response_data = df.dropna(subset=['first_response_hours'])
        if not response_data.empty:
            axes[1,1].hist(response_data['first_response_hours'], bins=15, edgecolor='black')
            axes[1,1].set_title('Response Time Distribution')
            axes[1,1].set_xlabel('Hours')
    
    plt.tight_layout()
    plt.show()

def main(json_file_path):
    """Main analysis function"""
    print("🎯 JIRA TICKET ANALYZER")
    print("=" * 50)
    
    # Load and process data
    print("Loading tickets...")
    tickets = load_jira_tickets(json_file_path)
    df = process_tickets(tickets)
    
    print(f"Loaded {len(df)} tickets\n")
    
    # Run all analyses
    metrics = {}
    metrics['issues'] = analyze_issues(df)
    print()
    
    metrics['response_times'] = analyze_response_times(df)
    print()
    
    metrics['sentiment'] = analyze_sentiment(df)
    print()
    
    metrics['satisfaction'] = calculate_satisfaction_score(df)
    print()
    
    # Create visualizations
    print("Creating charts...")
    create_charts(df, metrics)
    
    # Export results
    df.to_csv('analyzed_tickets.csv', index=False)
    print("\n✅ Analysis complete! Results saved to 'analyzed_tickets.csv'")
    
    return df, metrics

# Example usage
if __name__ == "__main__":
    # Replace with your JSON file path
    json_file = "your_jira_tickets.json"
    
    # Create sample data if file doesn't exist
    sample_data = [
        {
            "key": "PROJ-001",
            "summary": "Login system not working",
            "description": "Users unable to login to the application",
            "priority": "High",
            "status": "Resolved", 
            "created": "2024-01-10T09:00:00Z",
            "comments": [
                {"created": "2024-01-10T10:30:00Z", "body": "We are looking into this issue"},
                {"created": "2024-01-10T14:00:00Z", "body": "Issue has been fixed"}
            ]
        },
        {
            "key": "PROJ-002",
            "summary": "Feature request for dark mode",
            "description": "Users are requesting a dark theme option",
            "priority": "Low",
            "status": "Open",
            "created": "2024-01-11T11:00:00Z",
            "comments": [
                {"created": "2024-01-12T09:00:00Z", "body": "Thank you for the suggestion. We will consider this."}
            ]
        }
    ]
    
    # Save sample data if file doesn't exist
    try:
        with open(json_file, 'r') as f:
            pass
    except FileNotFoundError:
        print(f"Creating sample data file: {json_file}")
        with open(json_file, 'w') as f:
            json.dump(sample_data, f, indent=2)
    
    # Run analysis
    df, metrics = main(json_file)