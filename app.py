import csv
from pathlib import Path
import requests
import codecs
import streamlit as st
from llama_index import GPTVectorStoreIndex, ServiceContext, Document, StorageContext, load_index_from_storage
from llama_index import download_loader
from llama_index.llms import OpenAI
import openai

SimpleWebPageReader = download_loader("SimpleWebPageReader", custom_path="local_dir")
class ShiftJISToUTF8WebPageReader(SimpleWebPageReader):
    def load_data(self, urls):
        data = []
        for url in urls:
            response = requests.get(url)
            response.encoding = "Shift_JIS"
            content = response.content
            utf8_content = codecs.decode(content, "shift-jis", "ignore").encode("utf-8", "ignore").decode("utf-8")
            doc = Document(text=utf8_content)
            data.append(doc)
        return data

def is_shift_jis(response):
    content = response.text
    st.test("判定します")
    if 'charset=Shift_JIS' in content:
        st.text("Shift_jis変換します")
        return True
    else:
        return False

st.set_page_config(page_title="Chat with the Streamlit docs, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = st.secrets.OpenAIAPI.openai_api_key
st.title("教師ChatBotアプリ")

if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "こんにちは！私はクイズを出すChatBotです。何でも質問してください！"}
    ]
    
mode = st.radio(
    "質問モードと回答モードを切り替えてお使いください",
    ["***出題***", "***回答***"],
    horizontal = True)

if mode == "***回答***":
  @st.cache_resource(show_spinner=False)
  # チャットボットとやりとりする関数
  def load_data():
      with st.spinner(text="しばらくお待ちください"):
          documents = []
          with open('llamaindex_url.txt') as f:
              reader = csv.reader(f)
              for row in reader:
                  if row[0].endswith(".html"):
                      st.text("判定前")
                      if is_shift_jis(requests.get(row[0])):
                          loader = ShiftJISToUTF8WebPageReader()
                      else:
                          loader = SimpleWebPageReader()
                  else:
                      loader = SimpleWebPageReader()
                  documents.append(loader.load_data(urls = row)[0])
          service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="""
          {テーマ} = JR東日本の旅客営業規則 
          
          あなたは{テーマ}の専門家です。クライアントの質問に対して簡潔に説明し、それに関する4択の質問を出してください。
          """))
          st.text("うぇかぴぽ")
          index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
          return index


  index = load_data()

  if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
          st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

  if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
      st.session_state.messages.append({"role": "user", "content": prompt})

  for message in st.session_state.messages: # Display the prior chat messages
      with st.chat_message(message["role"]):
          st.write(message["content"])

  # If last message is not from assistant, generate a new response
  if st.session_state.messages[-1]["role"] != "assistant":
      with st.chat_message("assistant"):
          with st.spinner("考え中..."):
              response = st.session_state.chat_engine.chat(prompt)
              st.write(response.response)
              message = {"role": "assistant", "content": response.response}
              st.session_state.messages.append(message) # Add response to message history
else:
  st.text("きたよ")
