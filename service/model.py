from langchain.prompts import PromptTemplate

with open("prompts/main_prompt.txt", "r", encoding="utf-8") as f:
    prompt_text = f.read()

prompt = PromptTemplate(
    input_variables=["image_description"],
    template=prompt_text + "\n\n【图像描述】\n{image_description}"
)
