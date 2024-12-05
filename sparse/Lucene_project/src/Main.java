// Import necessary Lucene classes
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import javax.print.Doc;
import java.io.*;
import java.nio.file.Paths;
import java.util.List;
import java.util.Objects;

@SuppressWarnings("unused")
public class Main {

    public static void main(String[] args) {
        try {
            String directoryPath = "data/ft/all"; // Replace with your directory path
            List<FTDocumentParser.Document> documents = FTDocumentParser.parseDocuments(directoryPath);

            // System.out.println(documents.get(0));

            // get the document with specified docNo
            String docNo = "FT942-7623";
            FTDocumentParser.Document document = documents.stream()
                    .filter(doc -> Objects.equals(doc.getDocNo(), docNo))
                    .findFirst()
                    .orElse(null);
            
            if (document != null) {
                System.out.println(document);
            } else {
                System.out.println("Document with docNo " + docNo + " not found.");
            }

            System.out.println("Parsed " + documents.size() + " documents.");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
