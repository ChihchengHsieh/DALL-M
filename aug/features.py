import os, re, openai, torch, math
from typing import Optional
import pandas as pd

from enum import Enum
from loader.radiopaedia import RadioWebLoader
from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import TokenTextSplitter

from langchain.indexes import GraphIndexCreator
from langchain.graphs.networkx_graph import KnowledgeTriple
from langchain.llms import OpenAI
from langchain.chains import GraphQAChain
from langchain.graphs import Neo4jGraph
from secret import *


from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline, set_seed

from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain_core.documents import Document
from langchain.chains.qa_with_sources.retrieval
from tqdm import tqdm
from aug.graph_doc import (
    get_extraction_chain,
    data_to_graph_doc,
    chain_run,
    add_graph_documents,
)
from langchain.chat_models import ChatOpenAI
from langchain.chains import GraphCypherQAChain


openai.api_key = OPENAI_API_KEY

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)  # for exponential backoff


from aug.gpt import get_diagnosis

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


class DocumentSource(Enum):
    Wikipedia = "wikipedia"
    Radiopaedia = "radiopaedia"


class StoreType(Enum):
    Neo4jVectorIndex = "neo4jvectorindex"
    Neo4jGraph = "neo4jgraph"
    NetworkXGraph = "networkxgraph"


class QuestionLLM(Enum):
    ChatGPT = "chatgpt"
    Llama2 = "llama2"
    Mistral = "mistral"


CLEAN_QUERY = """
    MATCH (n)
    DETACH DELETE n
    """


def get_questions_by_lesion(lesion: str):
    questions = [
        # f"What is {lesion}?", # don't need the first one for extending features.
        f"What are the symptoms associated with {lesion}?",
        f"What can cause {lesion}?",
        f"What are the patient’s symptoms that are relevant for {lesion}?",
        f"What are the relevant clinical signs for the etiological diagnosis of {lesion}?",
        f"What are the relevant laboratory data for the etiological diagnosis of {lesion}?",
        f"What are the relevant clinical characteristics for the etiological diagnosis of {lesion}",
        f"What are the patient’s personal relevant history for the etiological diagnosis of {lesion}",
    ]
    return questions


def request_documents(
    lesion: str,
    sources: list[DocumentSource] = [
        DocumentSource.Radiopaedia,
        DocumentSource.Wikipedia,
    ],
    top1_only=True,
    text_splitter=TokenTextSplitter(chunk_size=2048, chunk_overlap=24),
):
    raw_documents = []
    # retrieve raw_documents
    if DocumentSource.Radiopaedia in sources:
        if top1_only:
            raw_documents.extend(RadioWebLoader(lesion, only_first=True).load())
        else:
            raw_documents.extend(RadioWebLoader(lesion, only_first=False).load())

    if DocumentSource.Wikipedia in sources:
        if top1_only:
            raw_documents.extend(WikipediaLoader(query=lesion, load_max_docs=1).load())
        else:
            raw_documents.extend(WikipediaLoader(query=lesion).load())

    # pre-process documents
    documents = text_splitter.split_documents(raw_documents)
    return documents


def clean_and_get_neo4jgraph(
    url: str = NOE4J_URL,
    username: str = NEO4J_USERNAME,
    password: str = NOE4J_PASSWORD,
):
    graph = Neo4jGraph(url=url, username=username, password=password)
    graph.query(CLEAN_QUERY)
    graph.refresh_schema()
    print(graph.schema)
    return graph


def networkx_get_responses(questions: list[str], documents: list[Document]):
    index_creator = GraphIndexCreator(llm=OpenAI(temperature=0))
    graph = None
    for d in documents:
        g_temp = index_creator.from_text(d.page_content)
        if graph is None:
            graph = g_temp
        else:
            triplets = graph.get_triples()
            for t in triplets:
                graph.add_triple(knowledge_triple=KnowledgeTriple(*t))
    chain = GraphQAChain.from_llm(OpenAI(temperature=0), graph=graph, verbose=True)
    res_dict = {}
    for q in questions:
        res = chain.run(q)
        res_dict[q] = res.strip()
    return res_dict


def neo4jvector_get_responses(questions: list[str], documents: list[Document]):
    _ = clean_and_get_neo4jgraph()
    db = Neo4jVector.from_documents(
        documents,
        OpenAIEmbeddings(),
        url=NOE4J_URL,
        username=NEO4J_USERNAME,
        password=NOE4J_PASSWORD,
    )
    retriever = db.as_retriever()
    # retriever = None
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k"),
        chain_type="stuff",
        retriever=retriever,
    )
    res_dict = {}
    for q in questions:
        res = chain(
            {"question": q},
            return_only_outputs=True,
        )
        res_dict[q] = res["answer"].strip()
    return res_dict


def neo4jgraph_get_responses(questions: list[str], documents: list[Document]):
    graph = clean_and_get_neo4jgraph()

    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)

    allowed_nodes = None
    allowed_rels = None
    # allowed_nodes = ["Symptom", "Disease"]
    # allowed_rels = ["CAN_CAUSE", "DESCRIBE", "HAS"]

    extract_chain = get_extraction_chain(llm, allowed_nodes, allowed_rels)
    gds = []

    for d in tqdm(documents, total=len(documents)):
        data = chain_run(extract_chain, d.page_content)
        # data = extract_chain.run(d.page_content)
        # graph_document = GraphDocument(
        #     nodes=[map_to_base_node(node) for node in data.nodes],
        #     relationships=[map_to_base_relationship(rel) for rel in data.rels],
        #     source=d,
        # )
        graph_document = data_to_graph_doc(data, d)
        # add_graph_document(graph, graph_document)
        gds.append(graph_document)

    graph = add_graph_documents(graph, gds)
    graph.refresh_schema()
    print(graph.schema)

    chain = GraphCypherQAChain.from_llm(
        graph=graph,
        cypher_llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k"),
        qa_llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k"),
        validate_cypher=True,  # Validate relationship directions
        verbose=True,
    )

    res_dict = {}
    for q in questions:
        try:
            res = chain.run(q)
            res_dict[q] = res.strip()
        except:
            res_dict[q] = "Generated Cypher Statement is not valid"
    return res_dict


def get_responses_from_documents(
    questions: list[str], documents: list[Document], store_type: StoreType
):
    if store_type == StoreType.NetworkXGraph:
        return networkx_get_responses(questions, documents)
    elif store_type == StoreType.Neo4jVectorIndex:
        return neo4jvector_get_responses(questions, documents)
    elif store_type == StoreType.Neo4jGraph:
        return neo4jgraph_get_responses(questions, documents)
    else:
        raise NotImplementedError(f"stor type {store_type} is not supported.")


@retry(
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(10),
)
def chat_completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def responses_to_keywords(lesion, responses: dict[str, str]):
    combined_response = " ".join(list(responses.values()))
    prompt = f"""For the subsequent paragraph, isolate solely those clinical keywords that are can be represented as symptoms or numerical values or boolean values (note: please separate the keywords by comma): 
    \"{combined_response}\"
    """
    res = chat_completion_with_backoff(
        # model="gpt-3.5-turbo",
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful clinical expert."},
            # {"role": "system", "content": "You are an experienced radiologist. Use a keyword-based report to answer the following questions."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        n=1,
    )
    keywords = [
        k.lower().strip()
        for k in res["choices"][0]["message"]["content"].replace(".", "").split(",")
        if not lesion in k.lower().strip()
    ]
    return set(keywords)


MIMIC_EYE_PATH = "F:\\mimic-eye"

REFLACX_LESION_LABEL_COLS = [
    # "Fibrosis",
    # "Quality issue",
    # "Wide mediastinum",
    # "Fracture",
    # "Airway wall thickening",
    ######################
    # "Hiatal hernia",
    # "Acute fracture",
    # "Interstitial lung disease",
    # "Enlarged hilum",
    # "Abnormal mediastinal contour",
    # "High lung volume / emphysema",
    # "Pneumothorax",
    # "Lung nodule or mass",
    # "Groundglass opacity",
    ######################
    "Pulmonary edema",
    "Enlarged cardiac silhouette",
    "Consolidation",
    "Atelectasis",
    "Pleural abnormality",
    # "Support devices",
]

CHEXPERT_LABEL_COLS = [
    "Atelectasis_chexpert",
    "Cardiomegaly_chexpert",
    "Consolidation_chexpert",
    "Edema_chexpert",
    "Enlarged Cardiomediastinum_chexpert",
    "Fracture_chexpert",
    "Lung Lesion_chexpert",
    "Lung Opacity_chexpert",
    "No Finding_chexpert",
    "Pleural Effusion_chexpert",
    "Pleural Other_chexpert",
    "Pneumonia_chexpert",
    "Pneumothorax_chexpert",
    "Support Devices_chexpert",
]


def get_report(
    data,
    mimic_eye_path: str = MIMIC_EYE_PATH,
    label_cols: str = REFLACX_LESION_LABEL_COLS,
    report_format=True,
):
    # reflacx_id = data['id']
    patient_id = data["subject_id"]
    study_id = data["study_id"]
    # dicom_id = data['dicom_id']
    report_path = os.path.join(
        mimic_eye_path,
        f"patient_{patient_id}",
        "CXR-DICOM",
        f"s{study_id}.txt",
    )
    with open(report_path) as f:
        report = f.read()

    report = report.strip().replace("FINAL REPORT\n", "").replace("\n", "").strip()

    age = data["age"]
    gender = "Female" if data["gender"] == "F" else "Male"
    if report_format:
        return re.sub(
            "[^0-9A-Za-z.\s\:']",
            "",
            f"{report} LESIONS:{get_diagnosis(data, label_cols)}. AGE: {age}. GENDER: {gender}.",
        )
    else:
        # return f"A {age} years old {gender} patient diagnosed with{get_diagnosis(data, label_cols)}. And, This patient has the radiology report: \n{report}\nThis patients is most likely to have {feature_to_name[desired_clinical_feature]} around"
        # return f"A {age} years old {gender} patient diagnosed with{get_diagnosis(data, label_cols)}. And, This patient has the radiology report: \n{report}\nThe {feature_to_name[desired_clinical_feature]} of this patient is around".replace("_", "")
        return re.sub(
            "[^0-9A-Za-z.\s\:']",
            "",
            f"A {age} years old {gender} patient diagnosed with{get_diagnosis(data, label_cols)}. And, This patient has the radiology report: \n{report}",
        )


def get_boolean_results_sys_p(
    report: str,
    identified_keywords: dict[str : list[str]],
    responses: Optional[dict[str, str]] = None,
):
    if not "boolean" in identified_keywords or len(identified_keywords["boolean"]) <= 0:
        return None

    sys_p = f"You are an experienced radiologist with more than 30 years of experience. You are the most respected radiologist in the world. \n\nYou are examining a patient with the following report:\n\n{report} \n\n=========\n"
    prompt = "Make a prediction based on your prior knowledge and on the patient's report. Please check if your prediction is in accordance with your prior knowledge. \n Ensure your answers are using following template.\n\n"
    if responses:
        sys_p += "According to your prior knowledge:\n\n"
        for q, a in responses.items():
            sys_p += f"{a}\n"
        sys_p += """=========\n\n"""
    # adjusted
    # sys_p += f"\n\n\n**Report:**\n=========\n{report}\n=========\n\n\nAccording to the report {mention_prior_knowledge}above, does the patient has the following symptoms/clinical signs/laboratory data/clinical characteristics/clinical history? (Return True or False only, and separate the answer for each attribute by comma.)\n"
    # original
    # prompt += f"\n\n\n**Report:**\n=========\n{report}\n=========\n\n\nAccording to the report {mention_prior_knowledge}above, does the patient has the following attributes? (Return True or False only, and separate the answer for each attribute by comma.)\n"

    tf_b = "{True\False}, because..."
    for i, k in enumerate(identified_keywords["boolean"]):
        prompt += f"{i+1}. {k}: {tf_b}\n"

    res = chat_completion_with_backoff(
        model="gpt-3.5-turbo",
        # model="gpt-4",
        messages=[
            # {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "system",
                "content": sys_p,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=1.2,
        # max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        n=1,
    )

    res_text = res["choices"][0]["message"]["content"].strip()

    values = []
    reasons = []

    for ans in [
        ans
        for ans in res_text.split("\n")
        if len(ans) > 5 and ans.split(".")[0].isdigit()
    ]:
        a = ans[3:]  # remove number
        # print(ans)
        comma_idx = a.index(",") if "," in a else float("inf")
        period_idx = a.index(".") if "." in a else float("inf")
        if math.isinf(comma_idx) and math.isinf(period_idx):
            idx_splitter = -1
        else:
            idx_splitter = min(comma_idx, period_idx)

        assert ":" in a, f"Not containing : in the answer: {ans}"
        after_boolean_index = a.index(":") + 2
        # assert (
        #     idx_splitter >= after_boolean_index
        # ), f"found idx_splitter {idx_splitter} >= after_boolean_index {after_boolean_index}, with answer: {ans}"

        boolean_ans = a[a.index(":") + 2 : idx_splitter]

        if idx_splitter == -1:
            reason = ""
        else:
            reason = a[idx_splitter + 1 :].strip()

        values.append(boolean_ans.strip().lower() == "true")
        reasons.append(reason)

    results = {k: v for k, v in zip(identified_keywords["boolean"], values)}
    reason_dict = {k: v for k, v in zip(identified_keywords["boolean"], reasons)}

    assert len(values) == len(
        identified_keywords["boolean"]
    ), f"""number of predicted values ({len(values)}) isn't the same as number of keywords ({len(identified_keywords['boolean'])})
    **Response**\n==========\n{res_text}\n
    **


    """

    return f"SYSTEM:\n\n{sys_p}\n\nUSER:\n\n{prompt}", res, results, reason_dict


def get_boolean_results(
    report: str,
    identified_keywords: dict[str : list[str]],
    responses: Optional[dict[str, str]] = None,
):
    if not "boolean" in identified_keywords or len(identified_keywords["boolean"]) <= 0:
        return None

    prompt = ""
    mention_prior_knowledge = ""
    if responses:
        prompt += f"**Prior Knowledge:**\n=========\n"
        for q, a in responses.items():
            prompt += f"Question: {q}\nAnswer: {a}\n"
        prompt += """=========\n"""
        mention_prior_knowledge = "and prior knowledge "
    # adjusted
    prompt += f"\n\n\n**Report:**\n=========\n{report}\n=========\n\n\nAccording to the report {mention_prior_knowledge}above, does the patient has the following symptoms/clinical signs/laboratory data/clinical characteristics/clinical history?\n"
    # original
    # prompt += f"\n\n\n**Report:**\n=========\n{report}\n=========\n\n\nAccording to the report {mention_prior_knowledge}above, does the patient has the following attributes? (Return True or False only, and separate the answer for each attribute by comma.)\nExample: False,False,True,True,False,False,True,True..."
    for i, k in enumerate(identified_keywords["boolean"]):
        prompt += f"{i+1}. {k}.\n"

    prompt += "Please ensure you answer all {len(identified_keywords['boolean'])} attributes with following template."

    tf_b = "{True\False}"
    for i, k in enumerate(identified_keywords["boolean"]):
        prompt += f"{i+1}. {tf_b}"

    res = chat_completion_with_backoff(
        model="gpt-3.5-turbo",
        # model="gpt-4",
        messages=[
            # {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "system",
                "content": "You are a medical expert.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
        n=1,
    )

    ##########
    res_text = res["choices"][0]["message"]["content"].strip()

    values = []

    for ans in [
        ans
        for ans in res_text.split("\n")
        if len(ans) > 5 and ans.split(".")[0].isdigit()
    ]:
        print(ans)
        a = ans[3:]  # remove number
        values.append(a.replace(",", "").replace(".", "").strip().lower() == "true")

    results = {k: v for k, v in zip(identified_keywords["boolean"], values)}

    assert len(values) == len(
        identified_keywords["boolean"]
    ), f"""number of predicted values ({len(values)}) isn't the same as number of keywords ({len(identified_keywords['boolean'])})
    **Response**\n==========\n{res_text}
    """
    #########
    # res_text = res["choices"][0]["message"]["content"].strip()
    # values = res_text.split(",")

    # assert len(values) == len(
    #     identified_keywords["boolean"]
    # ), f"""number of predicted values ({len(values)}) isn't the same as number of keywords ({len(identified_keywords['boolean'])})
    # **Response**\n==========\n{res_text}
    # """

    # results = {
    #     k: v.strip().replace(".", "") == "True"
    #     for k, v in zip(identified_keywords["boolean"], values)
    # }

    return prompt, res, results, None


def get_numerical_results(
    report: str,
    identified_keywords: dict[str : list[str]],
    responses: Optional[dict[str, str]] = None,
):
    if (
        not "numerical" in identified_keywords
        or len(identified_keywords["numerical"]) <= 0
    ):
        return None, None, None

    prompt = ""

    mention_prior_knowledge = ""
    if responses:
        prompt += f"**Prior Knowledge:**\n=========\n"
        for q, a in responses.items():
            prompt += f"Question: {q}\nAnswer: {a}\n"
        prompt += "=========\n"
        mention_prior_knowledge = "and prior knowledge "

    prompt += f"\n\n\n**Report:**\n=========\n{report}\n"
    for i, k in enumerate(identified_keywords["numerical"]):
        prompt += f"{k.upper()}: [MASK].\n"

    prompt += "=========\n\n\n"

    forcing = "Try to speculate the number instead of answering you don't know."
    prompt += f"According to the report {mention_prior_knowledge}above, please speculate be the numerical values covered by the token [MASK]? Please return only one single numerical values (not range) for each [MASK], and separate the answers by comma. {forcing}"

    res = chat_completion_with_backoff(
        # model="gpt-3.5-turbo",
        model="gpt-4",
        messages=[
            # {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "system",
                "content": "You are a medical expert.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
        n=1,
    )

    values = res["choices"][0]["message"]["content"].strip().split(",")
    values = [v.strip().replace(".", "") for v in values]
    values = [int(v) if v.isdecimal() else None for v in values]

    results = {k: v for k, v in zip(identified_keywords["numerical"], values)}

    return prompt, res, results


def get_numerical_results_sys_p(
    report: str,
    identified_keywords: dict[str : list[str]],
    responses: Optional[dict[str, str]] = None,
):
    if (
        not "numerical" in identified_keywords
        or len(identified_keywords["numerical"]) <= 0
    ):
        return None, None, None

    prompt = ""

    mention_prior_knowledge = ""
    if responses:
        prompt += f"**Prior Knowledge:**\n=========\n"
        for q, a in responses.items():
            prompt += f"Question: {q}\nAnswer: {a}\n"
        prompt += "=========\n"
        mention_prior_knowledge = "and prior knowledge "

    prompt += f"\n\n\n**Report:**\n=========\n{report}\n"
    for i, k in enumerate(identified_keywords["numerical"]):
        prompt += f"{k.upper()}: [MASK].\n"

    prompt += "=========\n\n\n"

    forcing = "Try to speculate the number instead of answering you don't know."
    prompt += f"According to the report {mention_prior_knowledge}above, please speculate be the numerical values covered by the token [MASK]? Please return only one single numerical values (not range) for each [MASK], and separate the answers by comma. {forcing}"

    res = chat_completion_with_backoff(
        # model="gpt-3.5-turbo",
        model="gpt-4",
        messages=[
            # {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "system",
                "content": "You are a medical expert.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
        n=1,
    )

    values = res["choices"][0]["message"]["content"].strip().split(",")
    values = [v.strip().replace(".", "") for v in values]
    values = [int(v) if v.isdecimal() else None for v in values]

    results = {k: v for k, v in zip(identified_keywords["numerical"], values)}

    return prompt, res, results


@retry(
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(10),
)
def get_possible_values(
    data: pd.Series,
    identified_keywords: dict[str, list[str]],
    responses: Optional[dict[str, str]] = None,
    using_sys_p: bool = True,
):
    report = get_report(data)

    # get_boolean_results
    (
        boolean_prompt,
        boolean_res,
        boolean_results,
        reason_dict,
    ) = (
        get_boolean_results_sys_p(report, identified_keywords, responses)
        if using_sys_p
        else get_boolean_results(report, identified_keywords, responses)
    )

    numerical_prompt, numerical_res, numerical_results = get_numerical_results(
        report, identified_keywords, responses
    )

    results = {}
    if boolean_res:
        results.update(boolean_results)
    if numerical_res:
        results.update(numerical_results)

    return (
        {
            "boolean": boolean_prompt,
            "numerical": numerical_prompt,
        },
        {
            "boolean": boolean_res,
            "numerical": numerical_res,
        },
        results,
        reason_dict,
    )


def chatgpt_questions_responses(questions: list[str]) -> dict[str, str]:
    res_dict = {}
    for q in questions:
        res = chat_completion_with_backoff(
            model="gpt-3.5-turbo",
            # model="gpt-4",
            messages=[
                # {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "system",
                    "content": "You are an experienced radiologist. Use a keyword-based report to answer the following questions.",
                },
                {"role": "user", "content": f"{q}"},
            ],
            temperature=0,
            n=1,
        )
        res_dict[q] = res
    return res_dict


def llama2_questions_responses(questions: list[str]) -> dict[str, str]:
    tokenizer = LlamaTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf", token=HUGGING_FACE_TOKEN
    )
    model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        token=HUGGING_FACE_TOKEN,
        device_map="auto",
        torch_dtype=torch.float16,
        eos_token_id=tokenizer.eos_token_id,
    )  # )
    # model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    set_seed(0)

    res_dict = {}
    for q in questions:
        res = generator(
            f"""Question: {q}\nAnswer:""",
            num_return_sequences=1,
            max_length=256,
            temperature=0,
            do_sample=False,
        )
        res_dict[q] = res[0]["generated_text"]
    return res_dict


def mistral_questions_responses(questions: list[str]) -> dict[str, str]:
    generator = pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-v0.1",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        pad_token_id=None,
        eos_token_id=2,
    )
    set_seed(0)

    res_dict = {}
    for q in questions:
        res = generator(
            f"""Question: {q}\nAnswer:""",
            num_return_sequences=1,
            max_length=256,
            temperature=0,
            do_sample=False,
        )
        res_dict[q] = res[0]["generated_text"]
    return res_dict


def LLM_get_responses(
    questions: list[str],
    llm: QuestionLLM,
) -> dict[str, str]:
    if llm == QuestionLLM.ChatGPT:
        return chatgpt_questions_responses(questions=questions)
    elif llm == QuestionLLM.Llama2:
        return llama2_questions_responses(questions=questions)
    elif llm == QuestionLLM.Mistral:
        return mistral_questions_responses(questions=questions)
    else:
        raise NotImplementedError(f"[{llm}] is not supported for generating responses.")
