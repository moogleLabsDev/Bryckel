# Bryckel
This platform allows users to upload, query, manage, and interact with PDF files. It provides functionalities for uploading PDFs, querying their content, reusing previously uploaded files, and managing uploaded files.

## Overview

1. **Upload PDF**:
   - Allow users to upload PDF files from their systems.

2. **Query PDF**:
   - Enable users to query the content of uploaded PDFs.

3. **Reuse Uploaded PDFs**:
   - Provide a selection list for previously uploaded PDFs to avoid re-uploading.

4. **Delete Incorrect Files**:
   - Offer an endpoint to delete mistakenly uploaded files.

## Core Features

1. **PDF Upload**:
   - Users can upload PDF files via the platform.
   - Uploaded files are stored and can be accessed for querying.

2. **Query Functionality**:
   - Users can query the content of uploaded PDFs using a simple interface.
   - The platform processes the query and returns relevant information from the document.

3. **File Reuse**:
   - A list of previously uploaded PDFs is available for users to select from.
   - This feature prevents the need for re-uploading the same document.

4. **File Management**:
   - Users can delete incorrect or unwanted files from the system.
   - This helps in maintaining a clean and relevant set of documents.


## Getting Started

### Prerequisites

- Python 3.8+
- Database setup 
- Cloud storage account 

### Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/moogleLabsDev/Bryckel.git
   cd Bryckel
   ```

2. **Backend Setup:**
   - **Create a virtual environment and activate it:**
     ```sh
     python -m venv venv
     source venv/bin/activate  # On Windows use `venv\Scripts\activate`
     ```
   - **Install backend dependencies:**
     ```sh
     pip install -r requirements.txt
     ```
   - **Set up environment variables:**
     Create a `.env` file in the root directory with necessary configuration, such as database credentials and storage details.
   - **Run the backend server:**
     ```sh
     uvicorn main:app  --reload
     ```
