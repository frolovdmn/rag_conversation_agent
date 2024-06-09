from typing import List, Tuple
from langchain_core.load import load
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

model = Ollama(model = 'qwen2')
embeddings = HuggingFaceEmbeddings()

memory = ConversationBufferMemory(
    memory_key = 'chat_history',
    output_key = 'answer',
    return_messages = True,
)

def create_retriever(file: str,
                     k: int):
    loader = PyPDFLoader(file)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, 
                                                   chunk_overlap = 150)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)

    retriever = db.as_retriever(search_type = 'similarity', 
                                search_kwargs = {'k': k})
    
    return retriever

file = '/Users/dmitriifrolov/Python/rag_conversation_agent/analysis.pdf'
chain_type = 'stuff'
k = 4

retriever = create_retriever(file, k)

def format_docs(docs: list) -> str:
    loaded_docs = [load(doc) for doc in docs]

    return '\n'.join(
        [
            f'<Document id={idx}>\n{doc.page_content}\n</Document>'
            for idx, doc in enumerate(loaded_docs)
        ]
    )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            'Ты - русскоязычный квалифицированный эндокринолог. Твоя задача - разобрать анализы и подсказать пациенту \
            какие показатели у него в норме, а над исправлением каких ему стоит поработать. Если человек спросит, что \
            значит аббревиатура - любезно расскажи ему. \
            Анализы пациента: \
            \n<Documents>\n{context}\n</Documents>'
        ),
        MessagesPlaceholder('chat_history'),
        (
            'human', 
            '{text}'
        )
    ]
)

class AgentInput(BaseModel):
    input: str = Field(default = 'Меня мучает бессонница, что мне может помочь?', 
                       description = 'Опишите вашу жалобу на здоровье?')
    chat_history: List[Tuple[str, str]] = Field(
        ..., 
        extra = {
            'widget': {'type': 'chat', 
                       'input': 'input' , 
                       'output': 'output'}
        }
    )

chain = (
    {
        'context': retriever | format_docs,
        'text': RunnablePassthrough(),
    }
    | prompt
    | model
    | StrOutputParser()
).with_types(input_type = AgentInput)