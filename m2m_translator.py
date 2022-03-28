from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# M2M100을 바로 이용한 기계번역 모델입니다.
model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_1.2B")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B")


def translate_to_eng(sen):
    tokenizer.src_lang = 'ko'
    encoded_ko = tokenizer(sen, return_tensors='pt')
    generated_tokens = model.generate(
        **encoded_ko, forced_bos_token_id=tokenizer.get_lang_id("en"))
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    print(f'origin: {sen}')
    print(f'translate: {result[0]}')
    return result[0]


def translate_to_ko(sen):
    tokenizer.src_lang = 'en'
    encoded_en = tokenizer(sen, return_tensors='pt')
    generated_tokens = model.generate(
        **encoded_en, forced_bos_token_id=tokenizer.get_lang_id("ko"))
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    #print(f'origin: {sen}')
    print(f'translate: {result[0]}')
    return result[0]
