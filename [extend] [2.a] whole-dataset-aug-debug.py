from aug.features import *
import warnings
import pandas as pd
from copy import deepcopy

warnings.filterwarnings("ignore")


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


top_5_lesions = [
    "pulmonary edema",
    "enlarged cardiac silhouette",
    "pulmonary consolidation",
    "atelectasis",
    "pleural abnormality",
]


responses = {
    "pulmonary edema": {
        "What are the symptoms associated with pulmonary edema?": "The symptoms associated with pulmonary edema include dyspnea (shortness of breath), hypoxemia (low oxygen levels), and respiratory failure. Other symptoms may include coughing, wheezing, frothy sputum, and chest pain. Diagnostic tests such as CBC, troponin, BNP, chest x-ray, CT scan, and ultrasound are used to diagnose and classify the cause of pulmonary edema. Treatment focuses on improving respiratory function, treating the underlying cause, and preventing further damage. Pulmonary edema can cause permanent organ damage and can lead to respiratory failure or cardiac arrest.",
        "What can cause pulmonary edema?": "Pulmonary edema can be caused by multiple factors, including cardiogenic causes (such as congestive heart failure, pericardial tamponade, heart valve dysfunction, and hypertensive crisis) and noncardiogenic causes (related to increased microvascular permeability). Various laboratory tests and imaging studies are used to diagnose and classify the cause of pulmonary edema. Treatment focuses on improving respiratory function, treating the underlying cause, and preventing further damage. Pulmonary edema can lead to respiratory failure or cardiac arrest if not treated promptly.",
        "What are the patient’s symptoms that are relevant for pulmonary edema?": "The relevant symptoms for pulmonary edema include dyspnea, hypoxemia, and respiratory failure. Other laboratory tests and imaging studies are used to diagnose and classify the cause of pulmonary edema. Treatment focuses on improving respiratory function, treating the underlying cause, and preventing further damage. Pulmonary edema can cause permanent organ damage and can lead to respiratory failure or cardiac arrest.",
        "What are the relevant clinical signs for the etiological diagnosis of pulmonary edema?": "The relevant clinical signs for the etiological diagnosis of pulmonary edema include dyspnea, hypoxemia, and respiratory failure. Various laboratory tests and imaging studies are often used to diagnose and classify the cause of pulmonary edema.",
        "What are the relevant laboratory data for the etiological diagnosis of pulmonary edema?": "Various laboratory tests (CBC, troponin, BNP, etc.) and imaging studies (chest x-ray, CT scan, ultrasound) are often used to diagnose and classify the cause of pulmonary edema.",
        "What are the relevant clinical characteristics for the etiological diagnosis of pulmonary edema": "The relevant clinical characteristics for the etiological diagnosis of pulmonary edema include impaired gas exchange, dyspnea, hypoxemia, and respiratory failure. Various laboratory tests and imaging studies are used to diagnose and classify the cause of pulmonary edema. Treatment focuses on improving respiratory function, treating the underlying cause, and preventing further damage. Pulmonary edema can cause permanent organ damage and can lead to respiratory failure or cardiac arrest. The classification of pulmonary edema includes cardiogenic and noncardiogenic causes. Cardiogenic pulmonary edema is typically caused by volume overload or impaired left ventricular function. Noncardiogenic causes are associated with increased microvascular permeability. Flash pulmonary edema is a particularly severe type of cardiogenic pulmonary edema.",
        "What are the patient’s personal relevant history for the etiological diagnosis of pulmonary edema": "The patient's personal relevant history for the etiological diagnosis of pulmonary edema would include information about their cardiac function, such as any history of congestive heart failure, pericardial tamponade, heart valve dysfunction, or hypertensive crisis. Other relevant information would include any factors that could contribute to increased hydrostatic pressure or impaired left ventricular function. Laboratory tests and imaging studies are often used to diagnose and classify the cause of pulmonary edema.",
    },
    "enlarged cardiac silhouette": {
        "What are the symptoms associated with enlarged cardiac silhouette?": "The symptoms associated with enlarged cardiac silhouette include sudden onset of sharp chest pain, pain in the shoulders, neck, or back, fever, weakness, palpitations, shortness of breath, dry cough, fatigue, anxiety, and diaphoresis. Other signs may include a friction rub heard during a cardiovascular examination, positional chest pain, pulsus paradoxus, low blood pressure, distant heart sounds, distension of the jugular vein, and signs of cardiac tamponade such as decreased alertness, pulsus paradoxus, low blood pressure, distant heart sounds, jugular vein distention, and equilibration of diastolic blood pressures. Complications can include pericardial effusion and cardiac tamponade.",
        "What can cause enlarged cardiac silhouette?": "Enlarged cardiac silhouette can be caused by pericarditis. Other possible causes include bacterial infections, heart attack, cancer, autoimmune disorders, and chest trauma.",
        "What are the patient’s symptoms that are relevant for enlarged cardiac silhouette?": "The relevant symptoms for an enlarged cardiac silhouette in pericarditis include sudden onset of sharp chest pain, which may also be felt in the shoulders, neck, or back. The pain is typically less severe when sitting up and more severe when lying down or breathing deeply. Other symptoms can include fever, weakness, palpitations, and shortness of breath.",
        "What are the relevant clinical signs for the etiological diagnosis of enlarged cardiac silhouette?": "The relevant clinical signs for the etiological diagnosis of enlarged cardiac silhouette include substernal or left precordial pleuritic chest pain with radiation to the trapezius ridge, relief of pain by sitting up or bending forward, worsening of pain by lying down or inspiration, dry cough, fever, fatigue, anxiety, friction rub heard on cardiovascular examination, positional chest pain, diaphoresis, pulsus paradoxus, low blood pressure, distant heart sounds, distension of the jugular vein, pericardial effusion, cardiac tamponade, and electrical alternans on EKG or Holter monitor.",
        "What are the relevant laboratory data for the etiological diagnosis of enlarged cardiac silhouette?": "The relevant laboratory data for the etiological diagnosis of enlarged cardiac silhouette in pericarditis include specific electrocardiogram (ECG) changes and fluid around the heart. These findings, along with the characteristic symptoms and physical examination findings, can help in diagnosing pericarditis.",
        "What are the relevant clinical characteristics for the etiological diagnosis of enlarged cardiac silhouette": "The relevant clinical characteristics for the etiological diagnosis of enlarged cardiac silhouette include substernal or left precordial pleuritic chest pain with radiation to the trapezius ridge, relief of pain by sitting up or bending forward, worsening of pain by lying down or inspiration, and the presence of a friction rub on cardiovascular examination. Complications of pericarditis can include pericardial effusion and cardiac tamponade.",
        "What are the patient’s personal relevant history for the etiological diagnosis of enlarged cardiac silhouette": "The patient's personal relevant history for the etiological diagnosis of enlarged cardiac silhouette includes symptoms such as sudden onset of sharp chest pain, which may also be felt in the shoulders, neck, or back, and is relieved by sitting up or bending forward. Other symptoms may include fever, weakness, palpitations, and shortness of breath. The cause of pericarditis is often unknown but can be due to viral or bacterial infections, heart attack, cancer, autoimmune disorders, and chest trauma. Diagnosis is based on the presence of chest pain, specific electrocardiogram (ECG) changes, and fluid around the heart. Treatment typically involves NSAIDs and possibly colchicine. Complications can include cardiac tamponade, myocarditis, and constrictive pericarditis.",
    },
    "pulmonary consolidation": {
        "What are the symptoms associated with pulmonary consolidation?": "The symptoms associated with pulmonary consolidation include reduced expansion of the thorax on inspiration, increased vocal fremitus, impaired percussion note, bronchial breath sounds, possible crackles, increased vocal resonance, and a lower PAO2 than calculated in the alveolar gas equation. Radiologically, an area of white lung is seen on X-ray.",
        "What can cause pulmonary consolidation?": "Pulmonary consolidation can be caused by accumulation of inflammatory cellular exudate in the alveoli and adjoining ducts, which can be pulmonary edema, inflammatory exudate, pus, inhaled water, or blood. It is also a characteristic sign of pneumonia.",
        "What are the patient’s symptoms that are relevant for pulmonary consolidation?": "The relevant symptoms for pulmonary consolidation include reduced expansion of the thorax on inspiration, increased vocal fremitus, impaired percussion note in the affected area, bronchial breath sounds, possible crackles, increased vocal resonance, and a lower PAO2 than calculated in the alveolar gas equation.",
        "What are the relevant clinical signs for the etiological diagnosis of pulmonary consolidation?": "The relevant clinical signs for the etiological diagnosis of pulmonary consolidation include reduced expansion of the thorax on inspiration, increased vocal fremitus, impaired percussion note in the affected area, bronchial breath sounds, possible crackles, increased vocal resonance, possible presence of a pleural rub, and a lower PAO2 than calculated in the alveolar gas equation. Radiologically, an area of white lung is seen on X-ray.",
        "What are the relevant laboratory data for the etiological diagnosis of pulmonary consolidation?": "The relevant laboratory data for the etiological diagnosis of pulmonary consolidation are not mentioned in the provided content.",
        "What are the relevant clinical characteristics for the etiological diagnosis of pulmonary consolidation": "The relevant clinical characteristics for the etiological diagnosis of pulmonary consolidation include reduced thorax expansion on the affected side, increased vocal fremitus on the affected side, impaired percussion note in the affected area, bronchial breath sounds, possible crackles, increased vocal resonance, and a lower PAO2 than calculated in the alveolar gas equation. Radiologically, an area of white lung is seen on X-ray, indicating consolidation.",
        "What are the patient’s personal relevant history for the etiological diagnosis of pulmonary consolidation": "The patient's personal relevant history for the etiological diagnosis of pulmonary consolidation is not provided in the given content.",
    },
    "pleural abnormality": {
        "What are the symptoms associated with pleural abnormality?": "The symptoms associated with pleural abnormality are pleural effusions and pleural thickening.",
        "What can cause pleural abnormality?": "Pleural abnormality can be caused by conditions such as pleural effusions, pneumothorax, and pleural thickening or plaques. Chest radiographs can be used to diagnose these conditions.",
        "What are the patient’s symptoms that are relevant for pleural abnormality?": "The relevant symptoms for pleural abnormality in a patient include pleural effusions and pleural thickening or plaques. These can be identified through a chest radiograph.",
        "What are the relevant clinical signs for the etiological diagnosis of pleural abnormality?": "The relevant clinical signs for the etiological diagnosis of pleural abnormality can be identified through a chest radiograph (chest X-ray). The chest radiograph can help diagnose conditions such as pneumonia, pneumothorax, interstitial lung disease, heart failure, bone fractures, hiatal hernia, and pulmonary tuberculosis. It is also used for screening job-related lung diseases. Additional imaging may be obtained to definitively diagnose or provide evidence for the suggested diagnosis based on the initial chest radiograph. The main regions where a chest X-ray may identify problems related to pleural abnormality include the costophrenic angles, which can indicate pleural effusions.",
        "What are the relevant laboratory data for the etiological diagnosis of pleural abnormality?": "The relevant laboratory data for the etiological diagnosis of pleural abnormality is a chest radiograph, also known as a chest X-ray (CXR). It is used to diagnose conditions affecting the chest, including pneumonia, pneumothorax, interstitial lung disease, heart failure, bone fractures, hiatal hernia, and pulmonary tuberculosis. Chest radiographs are commonly used in medicine and are the most common film taken. Different views of the chest, such as posteroanterior (PA), anteroposterior (AP), and lateral views, can be obtained to assess different aspects of the chest.",
        "What are the relevant clinical characteristics for the etiological diagnosis of pleural abnormality": "The relevant clinical characteristics for the etiological diagnosis of pleural abnormality can be identified through a chest radiograph (chest X-ray). Chest radiographs are used to diagnose conditions involving the chest wall, including the lungs, heart, and great vessels. They can help identify conditions such as pneumonia, pneumothorax, interstitial lung disease, heart failure, bone fractures, hiatal hernia, and pulmonary tuberculosis. Chest radiography is also used for screening job-related lung diseases. Additional imaging may be obtained to definitively diagnose a condition or provide evidence in favor of the initial diagnosis suggested by the chest radiograph.",
        "What are the patient’s personal relevant history for the etiological diagnosis of pleural abnormality": "The patient's personal relevant history for the etiological diagnosis of pleural abnormality is not mentioned in the provided content.",
    },
    "atelectasis": {
        "What are the symptoms associated with atelectasis?": "The symptoms associated with atelectasis may include cough (not prominent), chest pain (not common), breathing difficulty (fast and shallow), low oxygen saturation, pleural effusion (transudate type), cyanosis (late sign), and increased heart rate. However, atelectasis can also be asymptomatic. Fever is not a symptom of atelectasis.",
        "What can cause atelectasis?": "Atelectasis can be caused by various medical conditions, including post-surgical complications, surfactant deficiency, and poor surfactant spreading during inspiration. It can also be caused by blockage of a bronchiole or bronchus, such as by a foreign body, mucus plug, tumor, or compression from the outside. Risk factors for atelectasis include certain types of surgery, muscle relaxation, obesity, high oxygen, lower lung segments, age, chronic obstructive pulmonary disease (COPD), asthma, and type of anesthetic.",
        "What are the patient’s symptoms that are relevant for atelectasis?": "The relevant symptoms for atelectasis include cough (not prominent), chest pain (not common), breathing difficulty (fast and shallow), low oxygen saturation, pleural effusion (transudate type), cyanosis (late sign), and increased heart rate.",
        "What are the relevant clinical signs for the etiological diagnosis of atelectasis?": "The relevant clinical signs for the etiological diagnosis of atelectasis may include cough, chest pain (not common), breathing difficulty (fast and shallow), low oxygen saturation, pleural effusion (transudate type), cyanosis (late sign), and increased heart rate. However, it is important to note that atelectasis may also be asymptomatic.",
        "What are the relevant laboratory data for the etiological diagnosis of atelectasis?": "The relevant laboratory data for the etiological diagnosis of atelectasis are not provided in the given information.",
        "What are the relevant clinical characteristics for the etiological diagnosis of atelectasis": "The relevant clinical characteristics for the etiological diagnosis of atelectasis include cough (not prominent), chest pain (not common), breathing difficulty (fast and shallow), low oxygen saturation, pleural effusion (transudate type), cyanosis (late sign), and increased heart rate. It is important to note that atelectasis does not cause fever. The underlying causes of atelectasis can include adjacent compression, passive atelectasis, dependent atelectasis, and poor surfactant spreading. Risk factors for atelectasis include type of surgery, use of muscle relaxation, obesity, high oxygen, lower lung segments, age, presence of chronic obstructive pulmonary disease or asthma, and type of anesthetic. Diagnosis of atelectasis is generally confirmed through chest X-ray, which may show lung opacification and/or loss of lung volume. Additional imaging modalities such as CT chest or bronchoscopy may be necessary to determine the cause of atelectasis.",
        "What are the patient’s personal relevant history for the etiological diagnosis of atelectasis": "The patient's personal relevant history for the etiological diagnosis of atelectasis includes post-surgical atelectasis as a common cause, as well as pulmonary tuberculosis, smoking, and old age as risk factors. Other factors associated with the development of atelectasis include the presence of chronic obstructive pulmonary disease or asthma, and the type of anesthesia used. The diagnosis of atelectasis is generally confirmed through chest X-ray, which shows small volume linear shadows, usually peripherally or at the lung bases. CT chest or bronchoscopy may be necessary to determine the cause or confirm the absence of proximal obstruction.",
    },
}

identified_keywords = {
    "pulmonary consolidation": {
        "boolean": [
            "area of white lung on x-ray",
            "bronchial breath sounds",
            "impaired percussion note",
            "increased vocal fremitus",
            "increased vocal resonance",
            "lower pao2 than calculated in the alveolar gas equation",
            "possible crackles",
            "possible presence of a pleural rub",
            "reduced expansion of the thorax on inspiration",
        ],
        "numerical": [],
    },
    "pleural abnormality": {
        "boolean": [
            "bone fractures",
            "heart failure",
            "hiatal hernia",
            "interstitial lung disease",
            "plaques",
            "pleural effusions",
            "pleural thickening",
            "pneumonia",
            "pneumothorax",
            "pulmonary tuberculosis",
        ],
        "numerical": [],
    },
    "enlarged cardiac silhouette": {
        "boolean": [
            "anxiety",
            "back pain",
            "cardiac tamponade",
            "decreased alertness",
            "diaphoresis",
            "distant heart sounds",
            "distension of the jugular vein",
            "dry cough",
            "electrical alternans on ekg",
            "electrical alternans on holter monitor",
            "fatigue",
            "fever",
            "fluid around the heart",
            "friction rub",
            "left precordial pleuritic chest pain",
            "neck pain",
            "pain in the shoulders",
            "palpitations",  # can be heart-rate. (increased heart-rate)
            "pericardial effusion",
            "positional chest pain",
            "pulsus paradoxus",
            "radiation to the trapezius ridge",
            "relief of pain by bending forward",
            "relief of pain by sitting up",
            "shortness of breath",
            "specific electrocardiogram (ecg) changes",
            "substernal pain",
            "sudden onset of sharp chest pain",
            "weakness",
            "worsening of pain by inspiration",
            "worsening of pain by lying down",
        ],
        "numerical": [
            "systolic blood pressure (mmHg)",  # "low blood pressure"
            "diastolic blood pressure (mmHg)",  # "equilibration of diastolic blood pressures",
        ],
    },
    "pulmonary edema": {
        "boolean": [
            "chest pain",
            "congestive heart failure",
            "coughing",
            "dyspnea",
            "frothy sputum",
            "heart valve dysfunction",
            "hypertensive crisis",
            "hypoxemia",
            "impaired gas exchange",
            "impaired left ventricular function",
            "increased microvascular permeability",
            "pericardial tamponade",
            "respiratory failure",
            "volume overload",
            "wheezing",
        ],
        "numerical": [],
    },
    "atelectasis": {
        "boolean": [
            "anesthesia",
            "asthma",
            "asymptomatic",
            "breathing difficulty",
            "chest pain",
            "chronic obstructive pulmonary disease",
            "cough",
            "cyanosis",
            "fever",
            "pleural effusion",
            "pulmonary tuberculosis",
            "small volume linear shadows",
            "smoking",
        ],
        "numerical": [
            "heart rate (per minute)",  # increased heart-rate.
            "oxygen saturation (%)",  # reduced oxygen saturation.
        ],
    },
}

def main():
    # sample_df = pd.read_csv("./spreadsheets/gender-age-balance.csv") # for test purpose
    sample_df = pd.read_csv(
        "./spreadsheets/reflacx_clinical.csv"
    )  # for the whole dataset.

    for l in top_5_lesions:
        temp_df = deepcopy(sample_df)
        results = []
        all_res = []
        all_prompt = []
        all_reasons = []
        lesion_keywords = identified_keywords[l]
        lesion_responses = responses[l]
        for idx, data in temp_df.iterrows():
            prompt, res, result, reason = get_possible_values(
                data,
                lesion_keywords,
                responses=lesion_responses,
            )

            all_prompt.append(prompt)
            all_res.append(res)
            results.append(result)
            all_reasons.append(reason)

        for f in lesion_keywords["numerical"] + lesion_keywords["boolean"]:
            temp_df[f] = None
            temp_df[f] = [r[f] for r in results]

        temp_df["reason"] = all_reasons
        temp_df.to_csv(f"{l}-feature-extension.csv")


if __name__ == "__main__":
    main()
