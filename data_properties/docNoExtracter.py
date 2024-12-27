import os
import re

# Dizindeki dosyaları işleyecek kod
def extract_doc_numbers(input_dir, output_file):
    try:
        doc_numbers = []

        # Dizindeki tüm dosyaları gez
        for filename in os.listdir(input_dir):
            file_path = os.path.join(input_dir, filename)

            # Sadece dosyaları işle
            if os.path.isfile(file_path):
                with open(file_path, "r", encoding="utf-8") as infile:
                    content = infile.read()

                    # <DOCNO> etiketi içindeki numaraları bul
                    matches = re.findall(r"<DOCNO>(.*?)</DOCNO>", content)
                    doc_numbers.extend(matches)

        # Sonuçları çıktı dosyasına yaz
        with open(output_file, "w", encoding="utf-8") as outfile:
            for doc_no in doc_numbers:
                outfile.write(f"{doc_no}\n")

        print(f"Document numbers extracted to {output_file}.")

    except Exception as e:
        print(f"An error occurred: {e}")

# Kullanım örneği
input_directory = "CEng-778-Term-Project/data/ft/all"  # Dizin adı (değiştirin)
output_filename = "doc_numbers.txt"  # Çıktı dosyasının adı
extract_doc_numbers(input_directory, output_filename)

