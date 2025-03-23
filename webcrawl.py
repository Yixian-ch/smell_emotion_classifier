import requests
import json

def get_json_data(api_url):
    response = requests.get(api_url)
    data = response.json()
    return data

url = "https://explorer.odeuropa.eu/api/search?filter_emotion=http://data.odeuropa.eu/vocabulary/plutchik/trust&filter_language=fr&hl=en&page=1&sort=&type=smells"
json_data = get_json_data(url)

with open("output.json", "w", encoding="utf-8") as f:
    json.dump(json_data, f, ensure_ascii=False, indent=2)

def extract_data(json_path):
    final_data = []
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    for doc_id, result in enumerate(d["results"], start=1):
        source_dict = result.get("source", {})
        emotion_list = result.get("emotion", [])
        title = source_dict.get("label", None)
        source = source_dict.get("url", None)
        
        if len(emotion_list) > 0:
            emotion = emotion_list[0].get("label", None)
        else:
            emotion = None
        
        text = result.get("text", None)
        
        final_data.append({
            "doc_id": doc_id,
            "title": title,
            "source": source,
            "emotion": emotion,
            "text": text
        })
    
    return final_data
final_data = extract_data("output.json")


with open("final_output.json", "w", encoding="utf-8") as f:
    json.dump(final_data, f, ensure_ascii=False, indent=2)