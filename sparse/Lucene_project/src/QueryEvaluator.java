import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.queryparser.classic.QueryParserBase;
import org.apache.lucene.search.*;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import java.io.*;
import java.nio.file.Paths;
import java.util.*;

public class QueryEvaluator {

    public static class QueryInfo {
        String queryId;
        String queryText;

        QueryInfo(String queryId, String queryText) {
            this.queryId = queryId;
            this.queryText = queryText;
        }
    }

    public static List<QueryInfo> parseQueries(List<String> queryFilePaths) throws IOException {
        List<QueryInfo> queries = new ArrayList<>();

        for (String queryFilePath : queryFilePaths) {
            try (BufferedReader reader = new BufferedReader(new FileReader(queryFilePath))) {
                String line;
                String queryId = null;
                String queryText = null;

                while ((line = reader.readLine()) != null) {
                    if (line.startsWith("<num>")) {
                        queryId = line.replaceAll("<num> Number: ", "").trim();
                    } else if (line.startsWith("<title>")) {
                        queryText = line.replaceAll("<title>", "").trim();
                        if (queryId != null && queryText != null) {
                            queries.add(new QueryInfo(queryId, queryText));
                            queryId = null; // Reset queryId for the next query
                            queryText = null; // Reset queryText for the next query
                        }
                    }
                }
            }
        }

        return queries;
    }


    public static Map<String, Map<String, Integer>> parseQrels(List<String> qrelFilePaths) throws IOException {
        Map<String, Map<String, Integer>> qrels = new HashMap<>();

        for (String qrelFilePath : qrelFilePaths) {
            try (BufferedReader reader = new BufferedReader(new FileReader(qrelFilePath))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    if (line.isEmpty()) continue;
                    String[] parts = line.split(" ");
                    if (parts.length < 4) {
                        System.err.println("Skipping malformed line: " + line);
                        continue;
                    }
                    String queryId = parts[0];
                    String docId = parts[2];
                    int relevance = Integer.parseInt(parts[3]);

                    if (!docId.startsWith("FT")) continue; // We have only FT docs

                    qrels.putIfAbsent(queryId, new HashMap<>());
                    qrels.get(queryId).put(docId, relevance);
                }
            }
        }

        return qrels;
    }


    public static void evaluateQueries(String indexPath, List<QueryInfo> queries,
                                       Map<String, Map<String, Integer>> qrels,
                                       CharArraySet stopSet) throws Exception {
        Directory indexDir = FSDirectory.open(Paths.get(indexPath));
        try (DirectoryReader reader = DirectoryReader.open(indexDir)) {
            IndexSearcher searcher = new IndexSearcher(reader);
            Analyzer analyzer = new StandardAnalyzer(stopSet);
            double total_ndcg = 0;
            double total_recall = 0;
            double total_precision = 0;
            for (QueryInfo query : queries) {
                // Escape special characters in the query text
                String escapedQueryText = QueryParserBase.escape(query.queryText);

                QueryParser parser = new QueryParser("text", analyzer);
                Query luceneQuery = parser.parse(escapedQueryText);
                TopDocs results = searcher.search(luceneQuery, 10);
                ScoreDoc[] hits = results.scoreDocs;

                System.out.println("Query ID: " + query.queryId + " | Text: " + query.queryText);

                int tp = 0; // True Positives
                int fp = 0; // False Positives
                int fn = 0; // False Negatives

                int retrievedCount = hits.length;
                Map<String, Integer> queryQrels = qrels.getOrDefault(query.queryId, new HashMap<>());

                // Total number of relevant documents
                int relevantCount = (int) queryQrels.values().stream().filter(rel -> rel > 0).count();

                // List to store relevance scores of retrieved documents
                List<Integer> relevanceScores = new ArrayList<>();
                for (ScoreDoc hit : hits) {
                    Document doc = searcher.storedFields().document(hit.doc);
                    String docId = doc.get("docNo");

                    if (!queryQrels.containsKey(docId)) {
                        relevanceScores.add(0); // Not relevant
                        fp++;
                    } else {
                        int relevance = queryQrels.get(docId);
                        relevanceScores.add(relevance);
                        if (relevance > 0) {
                            tp++;
                        } else {
                            fp++;
                        }
                    }
                }

                // Calculate False Negatives
                fn = relevantCount - tp;

                // Precision and Recall calculation
                double precision = (tp + fp == 0) ? 0 : (double) tp / (tp + fp);
                double recall = (tp + fn == 0) ? 0 : (double) tp / (tp + fn);
                double f1 = (recall == 0 || precision == 0) ? 0 : 2 * precision * recall / (recall + precision);

                // Calculate DCG
                double dcg = 0.0;
                for (int i = 0; i < relevanceScores.size(); i++) {
                    int rel = relevanceScores.get(i);
                    dcg += (double) rel / (Math.log(i + 2) / Math.log(2)); // log base 2
                }

                // Calculate IDCG
                List<Integer> idealRelevanceScores = new ArrayList<>(queryQrels.values());
                idealRelevanceScores.sort(Collections.reverseOrder()); // Sort in descending order
                double idcg = 0.0;
                for (int i = 0; i < Math.min(idealRelevanceScores.size(), hits.length); i++) {
                    int rel = idealRelevanceScores.get(i);
                    idcg += (double) rel / (Math.log(i + 2) / Math.log(2)); // log base 2
                }

                // Normalize DCG to compute NDCG
                double ndcg = (idcg == 0) ? 0 : dcg / idcg;

                System.out.printf("Precision@10: %.2f | Recall@10: %.2f | F1@10: %.2f | NDCG@10: %.2f%n",
                        precision, recall, f1, ndcg);
                System.out.println("-------------------------------------------------");
                total_recall += recall;
                total_ndcg += ndcg;
                total_precision = precision;

            }
            double avg_prec = total_precision / queries.size();
            double avg_ndcg = total_ndcg / queries.size();
            double avg_recall = total_recall / queries.size();
            System.out.printf("Avg Precision@10: %.2f\n", avg_prec);
            System.out.printf("Avg recall@10: %.2f\n", avg_recall);
            System.out.printf("Avg ndcg@10: %.2f", avg_ndcg);
        }
    }


}
