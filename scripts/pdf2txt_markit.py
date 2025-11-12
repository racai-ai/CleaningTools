from markitdown import MarkItDown
import argparse
import os
from collections import defaultdict as dd 

info = dd(int)
sdict = {}
md = MarkItDown(enable_plugins=False)

def extract_text_from_pdf(file_path):
    try:
            pdf = md.convert(file_path)
            print(pdf.text_content)
            return pdf.text_content
    except Exception as e:
        print(f"Skipping {file_path} due to error: {e}")
        return ""

def count_words(text):
    return len(text.split())

def process_pdf_directory(directory_path, output_path):
    total_words = 0
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(directory_path, filename)
            text = extract_text_from_pdf(file_path)
            word_count = count_words(text)
            print(f"{filename}: {word_count} words")
            info[filename] = word_count
            total_words += word_count
            with open(os.path.join(output_path, filename + ".txt.md"), "w", encoding="utf-8") as f:
                f.write(text)
            
    print(f"\nTotal words in corpus: {total_words}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert PDFs in a directory to Markdown text files.")
    parser.add_argument("input_dir", help="Directory containing source PDF files.")
    parser.add_argument("output_dir", help="Directory where text outputs will be written.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    process_pdf_directory(args.input_dir, args.output_dir)
    sdict = dict(info)
