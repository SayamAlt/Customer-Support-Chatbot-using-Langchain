import requests
import json

url = "https://huggingface.co/datasets/MakTek/Customer_support_faqs_dataset/resolve/main/train_expanded.json"
response = requests.get(url)

if response.status_code == 200:
    try:
        lines = response.text.strip().split("\n")

        data = [json.loads(line) for line in lines]

        with open("customer_support_faqs.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(f"Dataset saved successfully as JSON. Loaded {len(data)} JSON objects.")

    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)

else:
    print(f"Failed to fetch dataset. HTTP Status Code: {response.status_code}")
