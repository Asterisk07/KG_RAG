# first line: 259
@memory.cache
def get_Gemini_response(instruction, system_prompt, temperature=0.0):
    res = fetch_Gemini_response(instruction, system_prompt, temperature)
    return res
