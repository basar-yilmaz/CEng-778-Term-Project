# Kodunuzu dosya adına göre çalıştırmak için uygun şekilde tanımlayabilirsiniz.
input_file = "qrels.txt"  # Girilen dosyanın adı
output_file = "query_relevance_count.txt"  # Çıktı dosyasının adı

target_prefixes = ["FT911", "FT921", "FT922", "FT923", "FT924", "FT931", "FT932", "FT933", "FT934", "FT941", "FT942", "FT943", "FT944"]  # Hedef döküman ön ekleri

# İçeriği işleme kodu
def count_query_relevance(input_file, output_file):
    try:
        query_relevance_count = {}

        with open(input_file, "r") as infile:
            for line in infile:
                parts = line.strip().split()
                # Format: query doc_id doc_name relevance
                if len(parts) == 4:
                    query, _, doc_name, relevance = parts
                    if relevance == "1" and any(doc_name.startswith(prefix) for prefix in target_prefixes):
                        if query not in query_relevance_count:
                            query_relevance_count[query] = 0
                        query_relevance_count[query] += 1

        # Sonuçları dosyayı yaz
        with open(output_file, "w") as outfile:
            for query, count in sorted(query_relevance_count.items()):
                outfile.write(f"{query}: {count}\n")

        print(f"Relevance counts per query written to {output_file}.")

    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Kodunuzu çalıştırın
count_query_relevance(input_file, output_file)

