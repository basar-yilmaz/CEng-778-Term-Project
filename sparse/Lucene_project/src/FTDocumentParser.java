import java.io.*;
import java.util.*;

public class FTDocumentParser {

    public static class Document {
        String docNo;
        String profile;
        String date;
        String headline;
        String text;
        String pub;
        String page;

            // Getters and setters 
        public String getDocNo() {
            return docNo;
        }

        public String getText() {
            return text;
        } 

        @Override
        public String toString() {
            return "Document{" +
                    "docNo='" + docNo + '\'' +
                    ", headline='" + headline + '\'' +
                    ", text='" + text + '\'' +
                    '}';
        }
    }

    public static List<String> parseStopwords(String stopwordsPath) throws IOException {
        List<String> stopwords = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(stopwordsPath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (!line.isEmpty()) {
                    stopwords.add(line);
                }
            }
        }
        return stopwords;
    }

    private static String extractTagContent(BufferedReader reader, String currentLine, String startTag, String endTag) throws IOException {
        StringBuilder content = new StringBuilder();
        
        // Process current line first
        if (currentLine.contains(startTag) && currentLine.contains(endTag)) {
            int startPos = currentLine.indexOf(startTag) + startTag.length();
            int endPos = currentLine.indexOf(endTag);
            return currentLine.substring(startPos, endPos).trim();
        }
        
        // Handle multi-line or partial line case
        if (currentLine.contains(startTag)) {
            int startPos = currentLine.indexOf(startTag) + startTag.length();
            content.append(currentLine.substring(startPos)).append(" ");
            
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.contains(endTag)) {
                    content.append(line.substring(0, line.indexOf(endTag)));
                    break;
                }
                content.append(line).append(" ");
            }
        }
        
        return content.toString().trim();
    }

    public static List<Document> parseDocuments(String directoryPath) throws IOException {
        List<Document> documents = new ArrayList<>();
        File folder = new File(directoryPath);
        File[] files = folder.listFiles();

        if (files != null) {
            for (File file : files) {
                if (file.isFile()) {
                    BufferedReader reader = new BufferedReader(new FileReader(file));
                    String line;
                    Document doc = null;

                    while ((line = reader.readLine()) != null) {
                        if (line.contains("<DOC>")) {
                            doc = new Document();
                        } else if (line.contains("</DOC>") && doc != null) {
                            documents.add(doc);
                            doc = null;
                        } else if (doc != null) {
                            if (line.contains("<DOCNO>")) {
                                doc.docNo = extractTagContent(reader, line, "<DOCNO>", "</DOCNO>");
                            } else if (line.contains("<PROFILE>")) {
                                doc.profile = extractTagContent(reader, line, "<PROFILE>", "</PROFILE>");
                            } else if (line.contains("<DATE>")) {
                                doc.date = extractTagContent(reader, line, "<DATE>", "</DATE>");
                            } else if (line.contains("<HEADLINE>")) {
                                doc.headline = extractTagContent(reader, line, "<HEADLINE>", "</HEADLINE>");
                            } else if (line.contains("<TEXT>")) {
                                doc.text = extractTagContent(reader, line, "<TEXT>", "</TEXT>");
                            } else if (line.contains("<PUB>")) {
                                doc.pub = extractTagContent(reader, line, "<PUB>", "</PUB>");
                            } else if (line.contains("<PAGE>")) {
                                doc.page = extractTagContent(reader, line, "<PAGE>", "</PAGE>");
                            }
                        }
                    }
                    reader.close();
                }
            }
        }
        return documents;
    }
}
