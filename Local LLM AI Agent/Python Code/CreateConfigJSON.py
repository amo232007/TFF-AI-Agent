import json
data = {
    "approved_departments": [
        "engineering", 
        "HR", 
        "marketing", 
        "sales", 
        "QA", 
        "accounting", 
        "business Development", 
        "production", 
        "warehousing"
    ]
}
#2. Open a file in write mode ('w') and save the data
with open("config.json", "w", encoding = "utf-8") as file:
    json.dump(data, file, indent = 4)
