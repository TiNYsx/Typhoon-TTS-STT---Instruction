from transformers import AutoModelForCausalLM, AutoTokenizer

def getAiResponse(userInput):
    messages = [
        {"role": "system", "content": "คุณคือผู้ช่วยที่จะตอบคำถามด้วยคำตอบที่ถูกต้อง สั้นแต่ได้ใจความ และเข้าใจง่าย."},
        {"role": "user", "content": userInput}
    ]

    input_ids = tokenizer_llm.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model_llm.device)

    terminators = [
        tokenizer_llm.eos_token_id,
        tokenizer_llm.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model_llm.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        temperature=0.7,
        top_p=0.95)
    response = tokenizer_llm.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return response

llmModel = "instruction/projects/llama3.2-typhoon2-3b-instruct"

tokenizer_llm = AutoTokenizer.from_pretrained(llmModel)
model_llm = AutoModelForCausalLM.from_pretrained(llmModel)

print(getAiResponse("สวัสดี, คุณช่วยสอนฉันทำข้าวผัดหน่อยได้มั้ย?"))