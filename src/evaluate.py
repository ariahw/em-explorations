import re


def extract_answer(answer) -> str:
    match = re.search(r'\\boxed\{([^}]*)\}', answer)
    if match:
        return match.group(1).strip()
    else:
        match = re.search(r'\\boxed\{([^}]*)\}', answer)
        if match:
            return match.group(1).strip()

    return None