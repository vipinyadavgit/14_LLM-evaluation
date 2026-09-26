# 14_LLM-evaluation

GOLDEN_DATASET
    |
Question
    |
Generator LLM:(Model1)
    | 
Generate answer + golden answer
    |
Judge llm(MODEL2)
    |
evaluation score
    |
dataframe format using pandas
    |
evaluation_results.csv

=============================================================================
RUN command
uncomment key and model in .env
uv run python main.py