## Controls the complete evaluator workflow.

import json
import pandas as pd

from generator_llm import generate_answer
from judge_llm import evaluate_answer

def run_evaluation():
    ##step 1: Load the dataset from the JSON file
    with open("data/goldendataset.json", "r") as f:
        dataset = json.load(f)

    result = []

    ###step 2:  process every question in the dataset.
    for item in dataset:

        question = item["question"]
        golden_answer = item["answer"]

        print("-"*50)
        print(f"\nQuestion: {question}")

        ##step 3: Generate an answer for the question using the first LLM.
        ##          Basically asks generator llm to generate an answer.
        
        generated_answer = generate_answer(question)

        print(f"\nGenerated Answer: {generated_answer}")

        ##step 4: Evaluate the generated answer against the golden answer using the second LLM.
        ##          Basically asks judge llm to evaluate the generated answer.
        
        evaluation = evaluate_answer(question, golden_answer, generated_answer)
        
        
        ##step 5: Print the evaluation results and store them in a list for later use.
        print("\n Judge evaluation:")
        print(evaluation)


        ###step 6: create one result record.
        row = {
            "question": question,
            "golden_answer": golden_answer,
            "generated_answer": generated_answer,
            "evaluation": evaluation
        }

        result.append(row)
        
     ####step 7: Save the evaluation results to a CSV file for further analysis. 
     ### Basically, convert result in to a dataframe. 
    df = pd.DataFrame(result)
    
    ## save it to a CSV file. 
    df.to_csv("data/evaluation_results.csv", index=False)

    print("\n")
    print("Evaluation completed")