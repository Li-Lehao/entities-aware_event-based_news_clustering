# Define input and output file paths
input_file = '../data/labeled_news_with_entities_60.csv'
output_file = '../data/labeled_news_with_entities_60_utf8.csv'

# Specify the current encoding (e.g., 'ISO-8859-1' or 'windows-1252')
current_encoding = 'ISO-8859-1'

# Read the file in the original encoding and write it in UTF-8
with open(input_file, 'r', encoding=current_encoding) as f_in:
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            f_out.write(line)

print(f'File encoding has been converted to UTF-8 and saved as {output_file}')
