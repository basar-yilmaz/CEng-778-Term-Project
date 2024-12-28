import os
import pickle


class Document:
    def __init__(
        self,
        doc_no=None,
        profile=None,
        date=None,
        headline=None,
        text=None,
        pub=None,
        page=None,
    ):
        self.doc_no = doc_no
        self.profile = profile
        self.date = date
        self.headline = headline
        self.text = text
        self.pub = pub
        self.page = page

    def __str__(self):
        return f"Document(doc_no='{self.doc_no}', headline='{self.headline}', text='{self.text}')"


class Query:
    def __init__(self, query_no=None, query=None, relevant_docs=None):
        self.query_no = query_no
        self.query = query
        self.number_of_relevant_docs = 0
        self.relevant_docs = relevant_docs

    def __str__(self):
        return f"Query(query_no='{self.query_no}', query='{self.query}', relevant_docs='{self.relevant_docs}')"

    def add_relevant_doc(self, doc_no):
        if self.relevant_docs is None:
            self.relevant_docs = []
        self.number_of_relevant_docs += 1
        self.relevant_docs.append(doc_no)

    def update_relevant_docs(self, relevant_docs):
        self.relevant_docs = relevant_docs

    def get_relevant_docs(self):
        return self.relevant_docs


def parse_relevance(file_paths, queries, doc_ids):
    processed_files = 0

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File not found - {file_path}")
            continue

        if not os.path.isfile(file_path):
            print(f"Warning: Not a file - {file_path}")
            continue

        print(f"Processing: {file_path}")
        processed_files += 1

        with open(file_path, "r") as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 4:
                    query_no, _, doc_no, relevance = parts
                    if relevance == "1" and doc_no in doc_ids:
                        for query in queries:
                            if query.query_no == query_no:
                                query.add_relevant_doc(doc_no)
                                break

    if processed_files == 0:
        raise FileNotFoundError("None of the provided paths were valid files")


def parse_queries(file_paths):
    queries = []
    for file_path in file_paths:
        if os.path.isfile(file_path):
            with open(file_path, "r") as file:
                lines = file.readlines()

            query = None
            for line in lines:
                line = line.strip()
                if "<top>" in line:
                    query = Query()
                elif "</top>" in line and query:
                    queries.append(query)
                    query = None
                elif query:
                    if "<num>" in line:
                        query.query_no = (
                            extract_tag_content([line], "<num>", "</num>")
                            .replace("Number: ", "")
                            .strip()
                        )
                    elif "<title>" in line:
                        query.query = extract_tag_content(
                            [line], "<title>", "</title>"
                        ).strip()
    return queries


def parse_stopwords(stopwords_path):
    stopwords = []
    with open(stopwords_path, "r") as file:
        for line in file:
            line = line.strip()
            if line:
                stopwords.append(line)
    return stopwords


def extract_tag_content(lines, start_tag, end_tag):
    content = []
    recording = False
    for line in lines:
        line = line.strip()
        if start_tag in line and end_tag in line:
            start = line.index(start_tag) + len(start_tag)
            end = line.index(end_tag)
            return line[start:end].strip()
        elif start_tag in line:
            start = line.index(start_tag) + len(start_tag)
            content.append(line[start:].strip())
            recording = True
        elif end_tag in line and recording:
            end = line.index(end_tag)
            content.append(line[:end].strip())
            break
        elif recording:
            content.append(line.strip())
    return " ".join(content)


def parse_documents(directory_path):
    documents = []
    doc_ids = set()

    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        if os.path.isfile(file_path):
            with open(file_path, "r") as file:
                lines = file.readlines()

            doc = None
            current_text = []
            inside_text = False  # Flag to track whether we're inside the <TEXT> tag
            for line in lines:
                line = line.strip()
                if "<DOC>" in line:
                    doc = Document()
                elif "</DOC>" in line and doc:
                    if current_text:
                        doc.text = " ".join(current_text).strip()
                        current_text = []
                    documents.append(doc)
                    doc = None
                elif doc:
                    if "<DOCNO>" in line:
                        doc.doc_no = extract_tag_content([line], "<DOCNO>", "</DOCNO>")
                        doc_ids.add(doc.doc_no)
                    elif "<PROFILE>" in line:
                        doc.profile = extract_tag_content(
                            [line], "<PROFILE>", "</PROFILE>"
                        )
                    elif "<DATE>" in line:
                        doc.date = extract_tag_content([line], "<DATE>", "</DATE>")
                    elif "<HEADLINE>" in line:
                        doc.headline = extract_tag_content(
                            [line], "<HEADLINE>", "</HEADLINE>"
                        )
                    elif "<TEXT>" in line:
                        inside_text = True
                        current_text.append(
                            extract_tag_content([line], "<TEXT>", "</TEXT>")
                        )
                    elif "</TEXT>" in line:
                        inside_text = False
                    elif inside_text:
                        current_text.append(line)
                    elif "<PUB>" in line:
                        doc.pub = extract_tag_content([line], "<PUB>", "</PUB>")
                    elif "<PAGE>" in line:
                        doc.page = extract_tag_content([line], "<PAGE>", "</PAGE>")

    return documents, doc_ids
