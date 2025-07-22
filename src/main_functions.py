def main_data_to_df():
    # Path to your CSV file
    csv_filepath = input("Enter the path to the plate data file: ")
    
    # Process the data
    result_df = process_plate_data(csv_filepath)
    
    # Display the result
    print("\nProcessed Data:")
    print(result_df.head(10))
    
    # Save to CSV
    result_df.to_csv("processed_plate_data.csv", index=False)
    print(f"Processed data saved to 'processed_plate_data.csv'")

def main_map_to_df():
    """Main entry point for the script."""
    csv_filepath = input("Enter the path to the plate map file: ")
    result_df = process_plate_map(csv_filepath)
    
    print("\nProcessed Map:")
    print(result_df.head(10))
    
    result_df.to_csv("processed_plate_map.csv", index=False)
    print("Processed map saved to 'processed_plate_map.csv'")

def main_protocol_to_df():
    # Path to your document file
    file_path = input("Enter the path to the document file (PDF, RTF, TXT, DOC, or DOCX): ")
    
    # Extract text from the file
    content = extract_text_from_file(file_path)
    
    # Send to Claude
    claude_response = send_to_claude(content)
    
    # Print the response
    print(claude_response)

    # Convert the response to a pandas dataframe
    df = pd.read_csv(claude_response)

    print("\nProcessed Protocol:")
    print(df.head(10))

    df.to_csv("processed_protocol.csv", index=False)
    print("Processed protocol saved to 'processed_protocol.csv'")
