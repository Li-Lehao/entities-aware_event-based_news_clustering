import numpy as np
import pandas as pd
import json

def generate_prompt_merge_entities():
    df = pd.read_csv('../data/labeled_news_with_entities_60_utf8.csv')
    all_entities = []
    for entities in df['entity']:
        all_entities.extend(entities.split(', '))
    # print(all_entities)

    prompt = f"""
        You are provided with a list of entity names and tasked with identifying and removing synonyms 
        (i.e., entity names that refer to the same entity).

        For example, "Singapore," "SG," "singapore," and "sg" all refer to the same country, Singapore, 
        and should be mapped to a single representative entity (e.g., "SG," or any one of them).

        Your task:
        1. Return a set called 'entities' containing the unique entity names after synonym removal.
        2. Return a dictionary called 'mapping' that maps each original entity name to its chosen representative in JSON format.
        Return only these two results in JSON format and nothing else.

        The given list of entity name: \n {all_entities}
    """

    print(prompt)
    return prompt
    
if __name__ == '__main__':
    generate_prompt_merge_entities()