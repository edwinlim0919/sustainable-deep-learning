import torch

# TOKENIZING THE INPUT
tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
text_1 = "Who was Jim Henson ?"
text_2 = "Jim Henson was a puppeteer"
indexed_tokens = tokenizer.encode(text_1, text_2, add_special_tokens=True)

# ENCODING INPUT SENTENCE IN A SEQUENCE OF LAST LAYER HIDDEN-STATES WITH BERTMODEL
segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
segments_tensors = torch.tensor([segments_ids])
tokens_tensor = torch.tensor([indexed_tokens])
model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')
with torch.no_grad():
    encoded_layers, _ = model(tokens_tensor, token_type_ids=segments_tensors)

# PREDICTING A MASKED TOKEN WTIH BERT MODELFORMASKEDLM
predict_masked = False

if predict_masked:
    masked_index = 8
    indexed_tokens[masked_index] = tokenizer.mask_token_id
    tokens_tensor = torch.tensor([indexed_tokens])
    masked_lm_model = torch.hub.load('huggingface/pytorch-transformers', 'modelForMaskedLM', 'bert-base-cased')
    with torch.no_grad():
        predictions = masked_lm_model(tokens_tensor, token_type_ids=segments_tensors)
    
    predicted_index = torch.argmax(predictions[0][0], dim=1)[masked_index].item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    print('PREDICT MASKED')
    print('predicted_index: ' + str(predicted_index))
    print('predicted_token: ' + str(predicted_token))

# ANSWERING QUESTIONS WITH BERT MODELFORQUESTIONANSWERING
answer_question = True

if answer_question:
    question_answering_model = torch.hub.load('huggingface/pytorch-transformers', 'modelForQuestionAnswering', 'bert-large-uncased-whole-word-masking-finetuned-squad')
    question_answering_tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-large-uncased-whole-word-masking-finetuned-squad')
    text_1 = "Jim Henson was a puppeteer"
    text_2 = "Who was Jim Henson ?"
    indexed_tokens = question_answering_tokenizer.encode(text_1, text_2, add_special_tokens=True)
    segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    segments_tensors = torch.tensor([segments_ids])
    tokens_tensor = torch.tensor([indexed_tokens])
    with torch.no_grad():
        out = question_answering_model(tokens_tensor, token_type_ids=segments_tensors)

    answer = question_answering_tokenizer.decode(indexed_tokens[torch.argmax(out.start_logits):torch.argmax(out.end_logits)+1])
    print('ANSWER QUESTION')
    print('answer: ' + str(answer))
