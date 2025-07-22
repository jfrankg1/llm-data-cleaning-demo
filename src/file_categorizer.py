import os
import base64
import json
import pandas as pd
import sys
from anthropic import Anthropic
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Initialize the Anthropic API
anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def prompt_for_multiple_files():
    """Prompt user to input file paths via command line"""
    print("Please enter the paths to your files (one per line).")
    print("Supported file types: CSV, TXT, Image (JPG, PNG), PDF, DOC, RTF")
    print("Enter an empty line when done.")
    
    valid_extensions = ['csv', 'txt', 'jpg', 'jpeg', 'png', 'pdf', 'doc', 'docx', 'rtf']
    files = []
    
    while True:
        file_path = input("\nEnter file path (or press Enter to finish): ").strip()
        
        if not file_path:  # Empty line
            break
            
        # Expand user's home directory if ~ is used
        file_path = os.path.expanduser(file_path)
        
        # Check if file exists
        if not os.path.isfile(file_path):
            print(f"Error: File not found: {file_path}")
            continue
            
        # Check file extension
        extension = file_path.split('.')[-1].lower()
        if extension in valid_extensions:
            files.append((file_path, extension))
            print(f"Added {file_path}")
        else:
            print(f"Error: Unsupported file type for {file_path}")
            print(f"Please use one of: {', '.join(valid_extensions)}")
    
    if not files:
        print("No files were selected.")
    else:
        print(f"\nSelected {len(files)} file(s):")
        for file_path, _ in files:
            print(f"  - {file_path}")
    
    return files

def main():
    # Call user to upload files
    files = prompt_for_multiple_files()
    
    if not files:
        print("No files to process. Exiting.")
        return
        
    # Categorize the files
    protocol_files, data_files, map_files, other_files = Claude_categorizer(files)

    # Print the results in a more organized way
    def print_category(category_name, file_list):
        if file_list:
            print(f"\n{category_name}:")
            for file in file_list:
                print(f"  - {file[0]}")
        else:
            print(f"\n{category_name}: None")

    print_category("Protocol files", protocol_files)
    print_category("Data files", data_files)
    print_category("Map files", map_files)
    print_category("Other files", other_files)

def Claude_categorizer(files):
    # This function will send the files to the Claude API (Haiku model to be tested first) to be categorized as data, protocol, or plate map
    # Uses Claude API to determine the type of file, and returns the type as a string
    # returns: "protocol", "data", "plate map"
    protocol_files = []
    data_files = []
    map_files = []
    other_files = []
    
    for file in files:
        file_content = process_file_for_claude(file[0])
        
        system_prompt = """You are a detail-oriented, precise, and patient researcher. Your long years of experience have taught you that doing the job properly the first time is more valuable than anything else, 
        so you do not guess. While some may even call you pedantic, everyone knows that you only make decisions in your job that are logical, rational, and supported by the evidence."""
        
        prompt = f"""
        You will be given a list of files along with their content and metadata. 
        Your task is to review this information and categorize each file into one of four categories based on its content and purpose. 
        Here is the list of files with their content and metadata:

        <file_list>
        {file_content}
        </file_list>

        Your task is to categorize each file into one of the following categories:
        1. "data" - for files containing experimental data
        2. "map" - for files mapping sample locations to sample identifiers
        3. "protocol" - for files specifying experimental protocol(s)
        4. "other" - for files containing some other type of information

        Instructions:
        1. Carefully review the content and metadata of each file in the list.
        2. Based on the information provided, determine which category best describes each file.
        3. Categorize each file into one and only one category.
        4. In cases where the categorization is ambiguous or unclear, use the "other" category.
        5. Do not guess or make assumptions about the file's content or purpose if it's not clearly evident from the provided information.

        Output your categorization as a JSON object, where each key is a filename and its value is the corresponding category. Use the following category labels exactly as written: "data", "map", "protocol", "other". Provide no other output.

        Example output format:
        {{
        "file1.txt": "data",
        "file2.csv": "map",
        "file3.docx": "protocol",
        "file4.pdf": "other"
        }}

        Remember:
        - Each file must be categorized into one and only one category.
        - If you're unsure about a file's category, use "other" rather than guessing.
        - Provide your final answer as a single JSON object inside <answer> tags.
        - Do not include any other text or comments in your response.
        """

        response_df = send_to_claude([file[0]], system_prompt, prompt)
        
        if response_df is not None and not response_df.empty:
            # Get the category for this file from the DataFrame
            file_category = response_df.iloc[0]['category']  # Assuming the category is in a column named 'category'
            
            if file_category == "protocol":
                protocol_files.append(file)
            elif file_category == "data":
                data_files.append(file)
            elif file_category == "map":
                map_files.append(file)
            else:
                other_files.append(file)
        else:
            # If we couldn't get a valid response, put the file in other_files
            print(f"Warning: Could not categorize {file[0]}, placing in other_files")
            other_files.append(file)
    
    return protocol_files, data_files, map_files, other_files

def process_file_for_claude(file_path):
    """Process a file for Claude API, returning its content and metadata"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except UnicodeDecodeError:
        # If text reading fails, read as binary and encode as base64
        with open(file_path, 'rb') as file:
            content = base64.b64encode(file.read()).decode('utf-8')
            is_binary = True
    else:
        is_binary = False
    
    extension = file_path.split('.')[-1].lower()
    mime_type_map = {
        'txt': 'text/plain',
        'csv': 'text/csv',
        'pdf': 'application/pdf',
        'doc': 'application/msword',
        'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png'
    }
    mime_type = mime_type_map.get(extension, 'application/octet-stream')
    
    return {
        'content': content,
        'mime_type': mime_type,
        'filename': os.path.basename(file_path),
        'is_image': mime_type.startswith('image/'),
        'is_binary': is_binary
    }

def send_to_claude(files, system_prompt, prompt, model="claude-3-5-haiku-20241022", max_tokens=1000, temperature=0.0):
    """Send files to Claude API and process the response"""
    # Build content array
    content = [{"type": "text", "text": prompt}]
    
    # Process each file
    for file_path in files:
        file_data = process_file_for_claude(file_path)
        
        if file_data['is_binary']:
            # For binary files, send as base64
            content.append({
                "type": "image" if file_data['is_image'] else "document",
                "source": {
                    "type": "base64",
                    "media_type": file_data['mime_type'],
                    "data": file_data['content']
                }
            })
        else:
            # For text files, send as plain text
            content.append({
                "type": "text",
                "text": f"File: {file_data['filename']}\nContent:\n{file_data['content']}"
            })
    
    # Send request to Claude
    message = anthropic.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": content}]
    )
    
    # Process response
    try:
        response_text = message.content[0].text
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            result = json.loads(json_str)
        else:
            result = json.loads(response_text)
            
        # Convert the result to a DataFrame with the correct structure
        df = pd.DataFrame([
            {'filename': filename, 'category': category}
            for filename, category in result.items()
        ])
        return df
    except Exception as e:
        print(f"Error parsing response: {str(e)}")
        print(f"Raw response: {response_text}")
        return None

if __name__ == "__main__":
    main()