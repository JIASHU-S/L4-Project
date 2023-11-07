import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
model.eval() 

# Define prediction function
def predict_proba(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')

    inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
    return probs

# Create LIME explainer
explainer = LimeTextExplainer(class_names=[f'class_{i}' for i in range(20)])  # Adjust the number of classes if necessary

# NOTE: Replace this with an actual example from the 20 newsgroups dataset
text_instance = "This is a sample text from the 20 newsgroups dataset to explain."

# Explain prediction
exp = explainer.explain_instance(text_instance, predict_proba, num_features=6)

# Display explanation
print('Explanation for the prediction:\n')
for feature, importance in exp.as_list():
    print(f'{feature}: {importance}')

