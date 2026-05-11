import csv
import sys

def csv_to_iob2(csv_file, output_file, text_col, label_col):
    """
    Convert CSV file to IOB2 format.
    
    Args:
        csv_file: Input CSV file path
        output_file: Output IOB2 file path
        text_col: Column name for tokens/text
        label_col: Column name for labels
    """
    with open(csv_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        reader = csv.DictReader(infile, delimiter=';')
        
        for row in reader:
            token = row[text_col].strip()
            label = row[label_col].strip()
            
            if token:
                outfile.write(f"{token} {label}\n")
            else:
                outfile.write("\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python csv_to_iob2.py <input.csv> <output.iob2> [text_col] [label_col]")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    output_file = sys.argv[2]
    text_col = sys.argv[3] if len(sys.argv) > 3 else "token"
    label_col = sys.argv[4] if len(sys.argv) > 4 else "label"
    
    csv_to_iob2(csv_file, output_file, text_col, label_col)
    print(f"Converted {csv_file} to {output_file}")