import re
import sys

def check_presentation(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 1. Check frame count
    frames = re.findall(r'\\begin\{frame\}(?:\[.*?\])?(?:\{([^}]*)\})?(.*?)\\end\{frame\}', text, re.DOTALL)
    print(f"[*] Total frames: {len(frames)}")

    # 2. Check balanced environments
    begins = re.findall(r'\\begin\{([a-zA-Z*]+)\}', text)
    ends = re.findall(r'\\end\{([a-zA-Z*]+)\}', text)
    if len(begins) != len(ends):
        print(f"[!] Mismatch in environment counts: {len(begins)} begins vs {len(ends)} ends!")
    else:
        print(f"[*] Environment counts perfectly match: {len(begins)} begins and ends.")

    # 3. Check balanced braces (strip true comments only)
    no_comments = re.sub(r'(?<!\\)%.*$', '', text, flags=re.MULTILINE)
    clean_braces = re.sub(r'\\\{|\\\}', '', no_comments)
    open_braces = clean_braces.count('{')
    close_braces = clean_braces.count('}')
    print(f"[*] Braces balance: {open_braces} open vs {close_braces} close (diff = {open_braces - close_braces}).")

    # 4. Check dollar signs
    dollars = no_comments.count('$')
    print(f"[*] Math dollar signs: {dollars} (even = {dollars % 2 == 0})")

    # 5. Check markdown artifacts
    if '**' in no_comments:
        print("[!] Warning: Markdown ** found in text!")
    else:
        print("[*] No markdown ** artifacts detected.")

    # 6. Inspect individual frames
    print("\n--- Frame-by-Frame Quality Check ---")
    for i, (title, body) in enumerate(frames, 1):
        display_title = title.strip() if title else "(Plain Frame)"
        
        # Check tabular
        tab_matches = re.findall(r'\\begin\{tabular\}\{(.*?)\}', body, re.DOTALL)
        tab_info = ""
        if tab_matches:
            tab_spec = tab_matches[0].strip()
            # extract widths
            widths = [float(w) for w in re.findall(r'[PC]\{([0-9.]+)cm\}', tab_spec)]
            if widths:
                tot = sum(widths)
                tot_padded = tot + len(widths) * 0.42
                status = "OK" if tot_padded <= 14.5 else "OVERFLOW"
                tab_info = f" [Table width: {tot:.1f}cm + padding = {tot_padded:.1f}cm -> {status}]"
            else:
                tab_info = f" [Table: {tab_spec}]"

        # Check images
        imgs = re.findall(r'\\includegraphics\[(.*?)\]\{([^}]+)\}', body)
        img_info = ""
        if imgs:
            img_info = f" [Images: {len(imgs)} -> {imgs[0][1]} ({imgs[0][0]})]"

        # Check items
        items = len(re.findall(r'\\item\b', body))
        item_info = f" [Items: {items}]" if items > 0 else ""

        print(f"Slide {i:02d}: {display_title[:45]:<45}{tab_info}{img_info}{item_info}")

if __name__ == "__main__":
    check_presentation("PPT/Team_15_Final_Review.tex")
