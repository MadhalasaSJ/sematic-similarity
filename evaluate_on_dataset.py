import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load dataset
df = pd.read_csv(r"C:\Users\madha\VisualStudio\semantic_similarity_project\DataNeuron_Text_Similarity.csv")

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Compute similarity
def compute_similarity(row):
    embeddings = model.encode([row['text1'], row['text2']])
    return util.cos_sim(embeddings[0], embeddings[1]).item()

df['similarity_score'] = df.apply(compute_similarity, axis=1)

# Save results
df.to_csv("evaluated_similarity_scores.csv", index=False)
print("Saved similarity scores to evaluated_similarity_scores.csv")
