import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.io.*;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public class Main {

    public static FTDocumentParser.Document getDocumentByDocNo(String docNo,
            List<FTDocumentParser.Document> documents) {
        return documents.stream()
                .filter(doc -> Objects.equals(doc.getDocNo(), docNo))
                .findFirst()
                .orElse(null);
    }

    public static List<FTDocumentParser.Document> createIndex(String directoryPath, String indexPath, CharArraySet stopSet) {
        List<FTDocumentParser.Document> documents = null;
        try {
            // Parse documents
            documents = FTDocumentParser.parseDocuments(directoryPath);
            System.out.println("Parsed " + documents.size() + " documents.");

            // Create the index
            Analyzer analyzer = new StandardAnalyzer(stopSet);
            IndexWriterConfig config = new IndexWriterConfig(analyzer);

            Directory indexDir = FSDirectory.open(Paths.get(indexPath));

            // Create an IndexWriter
            try (IndexWriter indexWriter = new IndexWriter(indexDir, config)) {
                for (FTDocumentParser.Document doc : documents) {
                    Document luceneDoc = new Document();
                    luceneDoc.add(new StringField("docNo", doc.getDocNo(), Field.Store.YES));
                    luceneDoc.add(new TextField("headline", doc.getHeadline() != null ? doc.getHeadline() : "", Field.Store.NO));
                    luceneDoc.add(new TextField("text", doc.getText() != null ? doc.getText() : "", Field.Store.NO));
                    luceneDoc.add(new TextField("date", doc.date != null ? doc.date : "", Field.Store.NO));
                    // luceneDoc.add(new TextField("profile", doc.profile != null ? doc.profile : "", Field.Store.NO));
                    // luceneDoc.add(new TextField("pub", doc.pub != null ? doc.pub : "", Field.Store.NO));
                    // luceneDoc.add(new TextField("page", doc.page != null ? doc.page : "", Field.Store.NO));

                    indexWriter.addDocument(luceneDoc);
                }

                indexWriter.commit();
            } catch (IOException e) {
                System.err.println("Error creating index: " + e.getMessage());
            }
        } catch (IOException e) {
            System.err.println("Error processing documents: " + e.getMessage());

        }
        return documents;
    }
    /*
    public static void main(String[] args) {
        String indexPath = "index_dir";
        String stopwordsPath = "data/ft/all/stopword.lst";

        // Get stopwords
        List<String> stopwords = null;
        try {
            stopwords = FTDocumentParser.parseStopwords(stopwordsPath);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        System.out.println("Number of stopwords: " + stopwords.size());
        CharArraySet stopSet = new CharArraySet(stopwords, true);

        //// Create index if not exists
        //List<FTDocumentParser.Document> docs = createIndex("data/ft/all", indexPath, stopSet);

        // Read index from disk
        Directory indexDir = null;
        try {
            indexDir = FSDirectory.open(Paths.get(indexPath));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        try (DirectoryReader reader = DirectoryReader.open(indexDir)) {
            // Create a searcher to query the index
            IndexSearcher searcher = new IndexSearcher(reader);

            Analyzer analyzer = new StandardAnalyzer(stopSet);

            // Define a query parser on a specific field, for example "text"
            QueryParser parser = new QueryParser("text", analyzer);
            Query query = parser.parse("Hubble Telescope Achievements");

            // Perform the search with a maximum number of hits
            ScoreDoc[] hits = searcher.search(query, 10).scoreDocs;

            for (ScoreDoc hit : hits) {
                // retrieve the document from the 'doc' field
                Document doc = searcher.storedFields().document(hit.doc);

//                // get docno from doc
//                String docNo = doc.get("docNo");
//                FTDocumentParser.Document dc = getDocumentByDocNo(docNo, docs);

                System.out.println(doc.get("docNo") + " : " + hit.score);
            }
        } catch (IOException | ParseException e) {
            throw new RuntimeException(e);
        }

    }
    */

    public static void main(String[] args) {
        String indexPath = "index_dir";

        List<String> queryFilePaths = Arrays.asList("data/query-relJudgments/q-topics-org-SET1.txt",
                                                "data/query-relJudgments/q-topics-org-SET2.txt",
                                                "data/query-relJudgments/q-topics-org-SET3.txt");
        
        String qrelFilePath = "data/query-relJudgments/qrel_301-350_complete.txt";

        List<String> qrelFilePaths = Arrays.asList("data/query-relJudgments/qrel_301-350_complete.txt",
                "data/query-relJudgments/qrels.trec7.adhoc_350-400.txt",
                "data/query-relJudgments/qrels.trec8.adhoc.parts1-5_400-450");

        String stopwordsPath = "data/ft/all/stopword.lst";

        //// Create index if not exists
        //List<FTDocumentParser.Document> docs = createIndex("data/ft/all", indexPath, stopSet);

        try {
            // Parse Stopwords
            List<String> stopwords = FTDocumentParser.parseStopwords(stopwordsPath);
            CharArraySet stopSet = new CharArraySet(stopwords, true);

            // Parse Queries
            List<QueryEvaluator.QueryInfo> queries = QueryEvaluator.parseQueries(queryFilePaths);

            // Parse Qrels
            Map<String, Map<String, Integer>> qrels = QueryEvaluator.parseQrels(qrelFilePaths);

            // Evaluate Queries
            QueryEvaluator.evaluateQueries(indexPath, queries, qrels, stopSet);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
