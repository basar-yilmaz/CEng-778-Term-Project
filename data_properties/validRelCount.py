import os
import re

# Kodunuz: Relevant document count belirlemek için.
def count_query_relevant_docs(relevant_file, available_docs_file, output_file):
    try:
        # Elimizde olan döküman listesini oku
        with open(available_docs_file, "r", encoding="utf-8") as infile:
            available_docs = set(line.strip() for line in infile)

        query_relevant_count = {}

        # Relevant query-döküman eşleşmelerini işle
        with open(relevant_file, "r", encoding="utf-8") as infile:
            for line in infile:
                parts = line.strip().split()
                if len(parts) == 4:
                    query, _, doc_name, relevance = parts
                    if relevance == "1":
                        if query not in query_relevant_count:
                            query_relevant_count[query] = 0
                        if doc_name in available_docs:
                            query_relevant_count[query] += 1

        # Sonuçları çıktı dosyasına yaz
        with open(output_file, "w", encoding="utf-8") as outfile:
            for query, count in sorted(query_relevant_count.items()):
                outfile.write(f"{query}: {count}\n")

        print(f"Query relevant document counts written to {output_file}.")

    except Exception as e:
        print(f"An error occurred: {e}")

# Kullanım örneği
relevant_file = "relevant_docs.txt"  # Relevant query-döküman dosyası
available_docs_file = "doc_numbers.txt"  # Elimizdeki dökümanların listesi
output_file = "valid_query_relevant.txt"  # Çıktı dosyası adı
count_query_relevant_docs(relevant_file, available_docs_file, output_file)

