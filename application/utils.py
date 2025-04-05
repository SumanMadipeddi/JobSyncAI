import re

def clean_text(text):
    text = re.sub(r'<[^>]*?>', '', text)  # Remove HTML tags
    text = re.sub(r'http[s]?://\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)  # Remove special characters
    text = re.sub(r'\s{2,}', ' ', text).strip()  # Remove extra spaces
    return text

def safe_format(value):
    if isinstance(value, dict):
        return "\n".join([f"- {k.capitalize()}: {v}" for k, v in value.items()])
    elif isinstance(value, list):
        return ", ".join(value)
    elif isinstance(value, str):
        return value.strip()
    else:
        return str(value)
