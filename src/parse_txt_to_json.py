import json

def parse_qa_output(raw_text):
    qa_dict = {}
    lines = raw_text.split("\n")
    current_q = None

    for line in lines:
        line = line.strip()
        if line.lower().startswith("q") and ":" in line:
            current_q = line.split(":", 1)[1].strip()
        elif line.lower().startswith("a") and ":" in line and current_q:
            answer = line.split(":", 1)[1].strip()
            qa_dict[current_q] = answer
            current_q = None
    return qa_dict

unit = input("Enter unit number (e.g., 1 or 2): ").strip()
filename = f"qa_unit{unit}_raw.txt"

try:
    with open(filename, "r", encoding="utf-8") as f:
        raw_text = f.read()

    qa_dict = parse_qa_output(raw_text)

    json_file = f"qa_unit{unit}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(qa_dict, f, indent=4, ensure_ascii=False)

    print(f" Parsed and saved updated Q&A to {json_file}")
except FileNotFoundError:
    print(f" Could not find file: {filename}")
