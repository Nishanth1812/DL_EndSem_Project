with open('src/pipeline.py', 'r', encoding='utf-8') as f:
    text = f.read()

import re
m = re.search(r'def build_data\(.*?\n(?=def )', text, re.DOTALL)
if m:
    print(m.group(0)[:1200])
