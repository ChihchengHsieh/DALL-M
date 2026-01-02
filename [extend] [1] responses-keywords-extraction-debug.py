from aug.features import *
import warnings
warnings.filterwarnings("ignore")

def get_questions_by_lesion(lesion: str):
    questions = [
        # f"What is {lesion}?", # don't need the first one for extending features.
        f"What are the symptoms associated with {lesion}?",
        f"What can cause {lesion}?",        
        f"What are the patient’s symptoms that are relevant for {lesion}?",
        f"What are the relevant clinical signs for the etiological diagnosis of {lesion}?",
        f"What are the relevant laboratory data for the etiological diagnosis of {lesion}?",
        f"What are the relevant clinical characteristics for the etiological diagnosis of {lesion}?",
        f"What are the patient’s personal relevant history for the etiological diagnosis of {lesion}?",
    ]
    return questions

def main():
    # top-5 lesions
    top_5_lesions = [
        "pulmonary edema",
        "enlarged cardiac silhouette",
        "pulmonary consolidation",
        "atelectasis",
        "pleural abnormality",
    ]

    response_dict = {}
    keyword_dict = {}

    for l in top_5_lesions:
        # adding prior knowledge from 8 questions.
        questions = get_questions_by_lesion(l)
        documents = request_documents(l, sources=[DocumentSource.Wikipedia])
        responses = neo4jvector_get_responses(
            questions, documents
        )  # let's try to get the responses from other LLMs to predict the keywords.
        keywords = responses_to_keywords(l, responses)

        response_dict[l] = responses
        keyword_dict[l] = keywords

    # For debugging purposes, you can print the dictionaries or save them to a file
    print("Responses:", response_dict)
    print("Keywords:", keyword_dict)

if __name__ == "__main__":
    main()