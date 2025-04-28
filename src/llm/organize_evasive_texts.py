import pandas as pd

def extract_rewritten_texts(input_file):
    df = pd.read_csv(input_file)
    
    marker = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"  # the EOT ID

    df['rewritten_text'] = df['text'].apply(lambda x: x.split(marker)[-1].strip() if marker in x else "")

    return df[['id', 'rewritten_text']]

def check_header_footer_issues(text):
    lines = text.split('\n')
    
    header_issue = False
    footer_issue = False

    if lines and lines[0].strip().endswith(':') and 'rewritten' in lines[0].strip().lower():
        lines = lines[1:]
        header_issue = True
    
    footer_keywords = ['note', 'rewrote', 'rewritten', 'ai']
    if lines:
        footer_line = lines[-1].strip().lower()
        keyword_count = sum(keyword in footer_line for keyword in footer_keywords)
        if keyword_count >= 2:
            lines.pop()
            footer_issue = True
    
    text = '\n'.join(lines).strip()
    
    return text, header_issue, footer_issue

def process_text_issues(df):
    df[['rewritten_text', 'header_issue', 'footer_issue']] = df['rewritten_text'].apply(lambda x: check_header_footer_issues(x)).apply(pd.Series)
    return df

def organize_texts():
    input_file = 'data/evasive_texts/raw/rephrased_texts_Control.csv'
    output_file = 'data/evasive_texts/control.csv'

    df = extract_rewritten_texts(input_file)
    df = process_text_issues(df)
    df = df[['id', 'header_issue', 'footer_issue', 'rewritten_text']]
    df.to_csv(output_file, index=False)

    print(f"Processed texts have been saved to {output_file}")

def get_text_statistics():
    files_to_check = [
        'data/evasive_texts/control.csv', 
        'data/evasive_texts/basic.csv', 
        'data/evasive_texts/advanced.csv'
    ]
    
    for file in files_to_check:
        df = pd.read_csv(file)

        num_empty_texts = df['rewritten_text'].isna().sum() + (df['rewritten_text'].str.strip() == '').sum()
        num_header_issues = df['header_issue'].sum()
        num_footer_issues = df['footer_issue'].sum()

        print(f"Statistics for {file}:")
        print(f"  Number of empty texts: {num_empty_texts}")
        print(f"  Number of header issues: {num_header_issues}")
        print(f"  Number of footer issues: {num_footer_issues}")
        print()
    
def remove_empty_entries():
    files_to_check = [
        'data/evasive_texts/control.csv', 
        'data/evasive_texts/basic.csv', 
        'data/evasive_texts/advanced.csv'
    ]
    
    for file in files_to_check:
        df = pd.read_csv(file)
        df = df.dropna(subset=['rewritten_text'])
        df.to_csv(file, index=False)
        print(f"Cleaned file saved to {file}")

if __name__ == "__main__":
    organize_texts()