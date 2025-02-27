import openai
 import time
 import pandas as pd
 import matplotlib.pyplot as plt
 import os
 from dotenv import load_dotenv
 
#Lade die API-Credentials aus der .env Datei

 load_dotenv()
 
#Setze die Azure OpenAI API-Variablen aus der .env Datei

 openai.api_type = "azure"

 openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")

 openai.api_version = os.getenv("AZURE_OPENAI_VERSION")

 openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
 
#Modell-Setup: Deployment-Namen aus der .env Datei laden

 test_models = [

     {"name": os.getenv("DEPLOYMENT_GPT4O"), "provider": "openai"},

     {"name": os.getenv("DEPLOYMENT_GPT4"), "provider": "openai"},

     {"name": os.getenv("DEPLOYMENT_GPT4_32K"), "provider": "openai"},

     {"name": os.getenv("DEPLOYMENT_GPT35_TURBO"), "provider": "openai"},

     {"name": os.getenv("DEPLOYMENT_GPT35_TURBO_16K"), "provider": "openai"},

 ]
 
#Test-Prompt

 prompt = "Wie kann ich mein Passwort zurücksetzen?"
 
benchmark_results = []
 
for model in test_models:

     if model["name"] is None:

         print(f"Fehler: Kein Deployment-Name für {model['provider']} gefunden. Überprüfe deine .env Datei!")

         continue  # Überspringe, falls kein Deployment-Name vorhanden ist
 
    start_time = time.time()

     try:

         response = openai.ChatCompletion.create(

             engine=model["name"],  #In Azure OpenAI muss "engine" verwendet werden!

             messages=[

                 {"role": "system", "content": "Du bist ein hilfreicher KI-Assistent."},

                 {"role": "user", "content": prompt}

             ]

         )

         end_time = time.time()
 
        answer = response["choices"][0]["message"]["content"]

         token_usage = response["usage"]["total_tokens"]

         latency = end_time - start_time
 
        benchmark_results.append({

             "Model": model["name"],

             "Antwort": answer[:1000] + "...",  #Nur die ersten 1000 Zeichen

             "Latenz (s)": latency,

             "Tokens": token_usage

         })
 
    except Exception as e:

         print(f"Fehler beim Aufruf von {model['name']}: {e}")
 
#Ergebnisse in DataFrame speichern

 df_results = pd.DataFrame(benchmark_results)

print(df_results)
 
#Visualisierung der Latenz-Zeiten

 plt.figure(figsize=(10, 5))

 plt.bar(df_results["Model"], df_results["Latenz (s)"], color='blue')

 plt.xlabel("Model")

 plt.ylabel("Latenz (s)")

 plt.title("Benchmark Latenzvergleich")

 plt.show()