{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Assistente legale virtuale\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Importazione librerie"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "from flask import Flask, render_template, request, jsonify\n",
    "from flask_ngrok import run_with_ngrok\n",
    "from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, load_index_from_storage\n",
    "from llama_index.storage.storage_context import StorageContext\n",
    "from llama_index.text_splitter import SentenceSplitter\n",
    "from llama_index.llms import OpenAI\n",
    "from llama_index.tools import QueryEngineTool, ToolMetadata\n",
    "from llama_index.agent import ReActAgent\n",
    "from utils import InitializeOpenAI\n",
    "import nest_asyncio\n",
    "import os"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Applicazione\n",
    "Per realizzare il nostro programma attraverso cui interagire col chatbot, abbiamo utilizzato **Flask**.\\\n",
    "Flask è un framework web open-source per Python che consente di creare applicazioni web in modo semplice ed efficiente.\\\n",
    "Il concetto generale attorno a cui funziona Flask è quello di **routing**, che consiste nell'associare determinate azioni a specifici URL. Si definiscono quindi delle **route** che indicano quale funzione o metodo deve essere eseguito quando viene raggiunto un determinato URL."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Innanzitutto abbiamo definito il nome dell'applicazione: *app*\\\n",
    "La funzione *run_with_ngrok*, messa a disposizione da Flask, permette di eseguire l'applicazione in un notebook."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "app = Flask(__name__)\n",
    "run_with_ngrok(app)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Il modulo **nest_asyncio** è progettato per risolvere problemi di compatibilità tra l'utilizzo di asyncio (Async I/O) e l'esecuzione di codice in ambienti che non sono specificamente progettati per supportare asyncio"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "nest_asyncio.apply()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Funzioni\n",
    "In questa sezione abbiamo definito le funzioni che vengono poi chiamate durante l'esecuzione dell'applicazione. Le funzioni create sono 3:\n",
    "- *SettingVectorIndices(topics_list, service_context)*\n",
    "- *LoadVectorIndex(topics_list, service_context)*\n",
    "- *IndividualQueryEngineTools(topic_list, index_list)*"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "*SettingVectorIndices(topics_list, service_context)* prende in input *topics_list*, che è una lista composta dalle stringhe corrispondenti agli argomenti trattati dai documenti che vogliamo indicizzare, e *service_context*. Quest'ultimo è un oggetto creato dalla funzione *ServiceContext()* e serve a dare al nostro modello la conoscenza di base della rete LLM da noi scelta.\\\n",
    "*ServiceContext()* è una classe messa a disposizione da LLamaindex per ottenere la base knowledge di una LLM e personalizzare il processo di indicizzazione dei documenti. Permette infatti di decidere quale LLM utilizzare, di scrivere un prompt che istruisca la rete per l'operazione di parsing, di specificare quale modello di embedding utilizzare e altro. L'output di SettingVectorIndices è un dizionario che racchiude i vector index dei due documenti."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "def SettingVectorIndices(topics_list, service_context):\n",
    "    doc_set = {}\n",
    "    all_docs = []\n",
    "    \n",
    "    for topic in topics_list:\n",
    "        reader = SimpleDirectoryReader(input_files=[f\"C:/Users/Raffa/Desktop/Text_mining/memoria/{topic}.txt\"])\n",
    "        topic_docs = reader.load_data()\n",
    "        # insert topic metadata into each topic\n",
    "        for d in topic_docs:\n",
    "            d.metadata = {\"topic\": topic}\n",
    "        doc_set[topic] = topic_docs\n",
    "        all_docs.extend(topic_docs)\n",
    "\n",
    "    index_set = {}\n",
    "    splitter = SentenceSplitter(separator=\"CIVIL CODE\")\n",
    "\n",
    "    for topic in topics_list:\n",
    "        nodes = splitter.get_nodes_from_documents(doc_set[topic])\n",
    "        cur_index = VectorStoreIndex.from_documents(\n",
    "            doc_set[topic],\n",
    "            service_context=service_context,\n",
    "            nodes = nodes,\n",
    "        )\n",
    "        index_set[topic] = cur_index\n",
    "        cur_index.storage_context.persist(persist_dir=f\"./storage/{topic}\")\n",
    "\n",
    "\n",
    "    return index_set"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "*LoadVectorIndex(topics_list, service_context)* è una funzione creata per quando si vogliono caricare gli indici già presenti nello storage e che permette quindi di evitare di utilizzare ad ogni sessione un LLM per ricostruire gli indici per degli specifici documenti."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "def LoadVectorIndex(topics_list, service_context):\n",
    "    index_set = {}\n",
    "    for topic in topics_list:\n",
    "        storage_context = StorageContext.from_defaults(\n",
    "            persist_dir=f\"./storage/{topic}\"\n",
    "        )\n",
    "        cur_index = load_index_from_storage(\n",
    "            storage_context = storage_context, service_context = service_context\n",
    "        )\n",
    "        index_set[topic] = cur_index\n",
    "\n",
    "    return index_set"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "*IndividualQueryEngineTools(topic_list, index_list)* prende in input, oltre a *topic_list*, *index_list* che non è altro che il return di SettingVectorIndices (o LoadVectorIndex). Questa funzione crea due tool messi a disposizione dell'agent. Un tool serve a ricercare testo pertinente a queries sulla spartizione dei beni tra divorzianti, analogamente l'altro tool per l'eredità."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "def IndividualQueryEngineTools(topic_list, index_list):\n",
    "\n",
    "    devorce = {'What does the civil code say about the obligations of spouses contracted before marriage?': 'As said in Article 211, the community property are liable for obligations contracted by one of the spouses before the marriage limited to the value of the property owned by that spouse before the marriage which, by agreement entered into, became part of the community of property.',\n",
    "           'If I divorce my spouse, when is community property to be considered terminated?': 'As said in the Article 191, in the case of legal separation, the community of property between the spouses is dissolved at the time when the president of the court authorises the spouses to live separately, or on the date of the signing of the minutes of the consensual separation of the spouses before the president, provided that they have been approved.',\n",
    "           'If my spouse squanders family property, can I cancel community property?': 'Yes, according to what is stated in Article 193 of the Civil Code. Provided that the conduct of the spouse engaged in the administration of property endangers the interests of the other or the community or family.'}\n",
    "    inheritance = {\"If I am entitled to a person's inheritance but die before accepting it, will my children be entitled to said inheritance?\": \"Yes, because as mentioned in Article 479 of the Civil Code, if the person called to the inheritance dies without accepting it, the right to accept it is passed on to the heirs.\"}\n",
    "    individual_query_engine_tools = []\n",
    "\n",
    "    devorce_examples = \"\"\n",
    "    inheritance_examples = \"\"\n",
    "\n",
    "    for query, answer in devorce.items():\n",
    "        devorce_examples += f\"{query}\\n{answer}\\n\\n\"\n",
    "\n",
    "    for query, answer in inheritance.items():\n",
    "        inheritance_examples += f\"{query}\\n{answer}\\n\\n\"\n",
    "\n",
    "    examples_list = [devorce_examples, inheritance_examples]\n",
    "\n",
    "    for i,topic in enumerate(topic_list):\n",
    "        query_engine = index_list[topic].as_query_engine()\n",
    "\n",
    "        metadata = ToolMetadata(\n",
    "                name=f\"vector_index_{topic}\",\n",
    "                description=f\"If you can, when answering questions regarding laws of the Civil Code also mention the number of the article you are referring to. Below are some examples.\\n\\n{examples_list[i]}\",\n",
    "            )\n",
    "\n",
    "        tool_instance = QueryEngineTool(query_engine=query_engine, metadata=metadata)\n",
    "        individual_query_engine_tools.append(tool_instance)\n",
    "\n",
    "\n",
    "    return individual_query_engine_tools"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Metodi dell'applicazione *app*\n",
    "Innanzitutto abbiamo definito una route per l'URL principale **'/'**. Quando l'applicazione Flask riceve una richiesta per questa route, eseguirà la funzione *index()* e restituirà il risultato di *render_template('index.html')*, cioè la pagina principale dell'applicazione.\\\n",
    "Dopo abbiamo definito una route per l'URL **/chat** che accetta solo richieste POST *(methods=['POST'])*.\n",
    "Quando l'applicazione Flask riceve una richiesta POST a questa route, eseguirà la funzione *chat()*.\n",
    "All'interno di chat(), si ottiene l'input dell'utente dalla richiesta POST: \n",
    "\n",
    "user_input = request.form['user_input']\n",
    "\n",
    "Quindi, utilizzando l'oggetto agent, viene chiamato il metodo chat() con l'input dell'utente per ottenere una risposta:\n",
    "\n",
    "response = agent.chat(user_input)\n",
    "\n",
    "La risposta ottenuta viene quindi estratta dal campo response dell'oggetto response: \n",
    "\n",
    "response_text = response.response.\n",
    "\n",
    "Infine, viene restituita una risposta JSON contenente il testo della risposta: \n",
    "\n",
    "return jsonify({'response': response_text})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "@app.route('/')\n",
    "def index():\n",
    "    return render_template('index.html')\n",
    "\n",
    "@app.route('/chat', methods=['POST'])\n",
    "def chat():\n",
    "    user_input = request.form['user_input']\n",
    "    response = agent.chat(user_input)\n",
    "\n",
    "    # Estrai il testo dalla risposta\n",
    "    response_text = response.response\n",
    "\n",
    "    return jsonify({'response': response_text})\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Main del codice\n",
    "Il codice mostrato sotto è il main che a sua volta chiama le funzioni viste finora a parte *InitializeOpenAI()* che è una funzione presente nel file *utils.py*. Ad ogni modo, InitializeOpenAI() definisce semplicemente la API key di OpenAI (di cui usufruiamo i servizi utilizzando il suo modello \"gpt-3.5-turbo-0613\" in ServiceContext e come response synthetizer) come variabile d'ambiente.\\\n",
    "Quindi, come si può vedere abbiamo utilizzato \"gpt-3.5-turbo-0613\" come modello per effettuare l'embedding e la sintesi delle risposte.\\\n",
    "L'if statement gestisce le due eventualità: caricare indexes presenti nello storage o rifarli daccapo."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " * Serving Flask app '__main__'\n",
      " * Debug mode: off\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.\n",
      " * Running on http://127.0.0.1:5000\n",
      "Press CTRL+C to quit\n",
      "127.0.0.1 - - [21/Dec/2023 12:49:22] \"GET / HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [21/Dec/2023 12:49:22] \"GET /favicon.ico HTTP/1.1\" 404 -\n",
      "Exception in thread Thread-4:\n",
      "Traceback (most recent call last):\n",
      "  File \"c:\\Jupyter\\envs\\Text_Mining\\Lib\\threading.py\", line 1038, in _bootstrap_inner\n",
      "    self.run()\n",
      "  File \"c:\\Jupyter\\envs\\Text_Mining\\Lib\\threading.py\", line 1394, in run\n",
      "    self.function(*self.args, **self.kwargs)\n",
      "  File \"c:\\Jupyter\\envs\\Text_Mining\\Lib\\site-packages\\flask_ngrok.py\", line 70, in start_ngrok\n",
      "    ngrok_address = _run_ngrok()\n",
      "                    ^^^^^^^^^^^^\n",
      "  File \"c:\\Jupyter\\envs\\Text_Mining\\Lib\\site-packages\\flask_ngrok.py\", line 38, in _run_ngrok\n",
      "    tunnel_url = j['tunnels'][0]['public_url']  # Do the parsing of the get\n",
      "                 ~~~~~~~~~~~~^^^\n",
      "IndexError: list index out of range\n",
      "127.0.0.1 - - [21/Dec/2023 12:51:05] \"POST /chat HTTP/1.1\" 200 -\n"
     ]
    },
    {
     "ename": "",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31mThe Kernel crashed while executing code in the the current cell or a previous cell. Please review the code in the cell(s) to identify a possible cause of the failure. Click <a href='https://aka.ms/vscodeJupyterKernelCrash'>here</a> for more info. View Jupyter <a href='command:jupyter.viewOutput'>log</a> for further details."
     ]
    }
   ],
   "source": [
    "if __name__ == \"__main__\":\n",
    "    InitializeOpenAI()\n",
    "    llm = OpenAI(model=\"gpt-3.5-turbo-0613\")\n",
    "    service_context = ServiceContext.from_defaults(llm=llm)\n",
    "\n",
    "    storage_path = \"storage\"\n",
    "    topics = ['DIVISION_OF_ASSETS_AFTER_DIVORCE', 'INHERITANCE']\n",
    "\n",
    "    if len(os.listdir(storage_path)) > 0:\n",
    "        index_set = LoadVectorIndex(topics, service_context)\n",
    "    else:\n",
    "        index_set = SettingVectorIndices(topics)\n",
    "\n",
    "    individual_query_engine_tools = IndividualQueryEngineTools(topics, index_set)\n",
    "\n",
    "    tools = individual_query_engine_tools\n",
    "    agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)\n",
    "\n",
    "    app.run()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Text_Mining",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
