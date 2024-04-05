from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
from langchain.vectorstores.azuresearch import AzureSearch
from cognitive_search import get_similar_content
from langchain.schema import Document

from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

import re


def get_model_response(user_query: str, vector_store: AzureSearch, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):

    docs: list[Document] = get_similar_content(user_query, vector_store)

    context = ""

    for doc in docs:
        context+=doc.page_content + '\n\n'

    
    system_message = f"""You are a helpful assistant that answers the given question, in english, based on the provided context by extracting the information from it. You also don't use your own training knowledge. The answer should be grammatically correct and explained in a step by step manner. --- Context here: {context}"""
    model_reminder = "If the question asked is not relevant to the context received below or you cannot answer, say politely that you can not answer. Do not use your own knowledge to answer the given question. Do not write anything else except the answer."


    prompt = [{'role': 'system', 'content': system_message},{'role': 'user', 'content': f'{model_reminder} Question: {user_query}'}]

    inputs = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        return_tensors='pt',
        tokenize=True,
    )

    tokens = model.generate(
        inputs.to(model.device),
        max_new_tokens=10000,
        temperature=0.8,
        do_sample=True
    )

    answer = tokenizer.decode(tokens[0], skip_special_tokens=False)

    answer = ''.join(answer.split('\n'))

    result = re.search('<\\|assistant\\|>(.*)<\\|endoftext\\|>', answer)
    print(result)
    print('\n')
    print(type(result))

    result = result.group(1)


    return result




# from transformers import AutoModelForCausalLM, AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained('stabilityai/stablelm-zephyr-3b')
# model = AutoModelForCausalLM.from_pretrained(
#     'stabilityai/stablelm-zephyr-3b',
#     device_map="auto"
# )

# prompt = [{'role': 'user', 'content': 'What is the meaning of life?'}]
# inputs = tokenizer.apply_chat_template(
#     prompt,
#     add_generation_prompt=True,
#     return_tensors='pt'
# )

# tokens = model.generate(
#     inputs.to(model.device),
#     max_new_tokens=1024,
#     temperature=0.8,
#     do_sample=True
# )

# print(tokenizer.decode(tokens[0], skip_special_tokens=False))