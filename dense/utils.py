def create_test_ids_from_qrels(qrels_path: str, output_path: str = "test_ids.txt") -> None:
    """
    Creates a text file containing unique query IDs from a .qrels file.
    
    Args:
        qrels_path (str): Path to the .qrels file
        output_path (str): Path where to save the test_ids.txt file
        
    Returns:
        None
    """
    # Set to store unique query IDs
    unique_qids = set()
    
    # Read the .qrels file and extract query IDs
    with open(qrels_path, 'r', encoding='utf-8') as f:
        for line in f:
            # .qrels format: qid 0 docid relevance
            parts = line.strip().split()
            if len(parts) >= 4:
                qid = parts[0]
                unique_qids.add(qid)
    
    # Write unique query IDs to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        for qid in sorted(unique_qids):
            f.write(f"{qid}\n")

if __name__ == "__main__":
    create_test_ids_from_qrels("test.qrels")