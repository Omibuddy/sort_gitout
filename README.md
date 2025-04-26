# sort_itout
finanz-informatik - AI hackathon 2025
## Pre-filter dataset and Modes for LLM 
# why?
/ Summarizes commits.
/ reads the code differences.
/ Stored for fast access.
/ AI chatbot answering your questions.
/ Understands the “why” of changes.

# Overall
Data Ingestion: Clone or pull the GitHub repository, extract raw code and resource files, and batch them into manageable chunks.
Overflow Filtering: Apply length and token-count checks (e.g., remove or truncate segments) to prevent buffer overflows or rate-limit errors.
LLM Feed & Mode Dispatch: Feed the cleaned data into the LLM and route outputs through one of five modes based on task:

Standard: General text responses
Visualization Analysis: Generate charts or plots
Errors & Components Analysis: Diagnose failures and inspect module internals

Tabular View: Present structured data tables
Code Mode: (Proposed) Return executable code snippets tailored to user queries
