target_prefixes = ["FT911", "FT921", "FT922", "FT923", "FT924", "FT931", "FT932", "FT933", "FT934", "FT941", "FT942", "FT943", "FT944"]  # Hedef döküman ön ekleri

# Kodunuzu dosya adına göre çalıştırmak için uygun şekilde tanımlayabilirsiniz.
input_file = "qrels.txt"  # Girilen dosyanın adı
output_file = "filtered_relevant_docs.txt"  # Çıktı dosyasının adı

# İçeriği işleme kodu
def filter_relevant_documents(input_file, output_file):
    try:
        with open(input_file, "r") as infile, open(output_file, "w") as outfile:
            relevant_lines = []
            for line in infile:
                parts = line.strip().split()
                # Format: query doc_id doc_name relevance
                if len(parts) == 4:
                    query, _, doc_name, relevance = parts
                    if any(doc_name.startswith(prefix) for prefix in target_prefixes):
                        relevant_lines.append(line)

            # Sonucunu yeni dosyayı yaz
            outfile.writelines(relevant_lines)
        print(f"{len(relevant_lines)} relevant lines written to {output_file}.")

    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Kodunuzu çalıştırın
filter_relevant_documents(input_file, output_file)
