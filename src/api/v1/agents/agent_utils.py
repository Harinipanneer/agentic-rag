

# FORMATTING OUTPUT
def format_llm_output(res):
    content = res.content

    # dict 
    if isinstance(content, dict):
        return content.get("text", str(content))

    # list
    if isinstance(content, list):
        cleaned = []
        for item in content:
            if isinstance(item, dict):
                cleaned.append(item.get("text", str(item)))
            else:
                cleaned.append(str(item))
        return " ".join(cleaned)

    return str(content)

