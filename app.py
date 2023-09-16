from langchain.llms import CTransformers
from langchain.chains import LLMChain
from langchain import PromptTemplate
import gradio as gr
import time

#load the model
code_llama_model = CTransformers(
    model = "codellama-7b-instruct.Q4_K_M.gguf",
    config={'max_new_tokens':512,"temperature":0.3}
)

#create template
prompt_template = """
You are an Expert AI Coding Assistant tasked with solving coding problems 
and providing code snippets based on the user's query.
Query: {query}

Here's a helpful code snippet:
"""
#Gradio interface
with gr.Blocks(title='Codellama-7b-instruct Demo') as codellama_demo:
    chatbot = gr.Chatbot([],elem_id="Chatbot",height=500)
    user_input= gr.Textbox()
    clear_button = gr.ClearButton([user_input,chatbot])

    def generate_response(query):
        prompt = PromptTemplate(template=prompt_template,input_variables=['query'])
        chain = LLMChain(prompt=prompt ,llm=code_llama_model)
        response = chain.run({'query':query})
        return response
    def chat_with_bot(message,chat_history):
        bot_message = generate_response(message)
        chat_history.append((message,bot_message))
        time.sleep(2)
        return "",chat_history
    
    user_input.submit(chat_with_bot,[user_input,chatbot],[user_input,chatbot])

codellama_demo.launch()
