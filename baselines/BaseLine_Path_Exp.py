import torch
import time
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import warnings
from util import text_util
from agatha.util.sqlite3_lookup import Sqlite3LookupTable
from openai_llm import oai_get_response

warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PathExplainer:
    def __init__(self, sents_db_path, model_name='all-MiniLM-L6-v2'):
        self.sents_db = Sqlite3LookupTable(sents_db_path)
        self.model_sbert = SentenceTransformer(model_name, device="cpu")
    
    def get_embedding(self, text):
        return self.model_sbert.encode(text, convert_to_tensor=True)
    
    def retrieve_abstracts(self, pmids):
        abstracts = []
        for pmid in tqdm(pmids, desc='Retrieving abstracts'):
            abstr_text = text_util.get_abstr_text(pmid, self.sents_db)
            abstracts.append(abstr_text)
        return abstracts
    
    def find_best_selection(self, source, target, edges_pmids):
        query = f"{source} {target}"
        query_embedding = self.get_embedding(query)
        best_selection = {}
        cosine_similarities = {}
        
        for edge, pmids in edges_pmids.items():
            if not pmids:
                print(f"No PMIDs for edge {edge}, skipping...")
                continue
            
            pmid_texts = [text_util.get_abstr_text(pmid, self.sents_db) for pmid in pmids]
            pmid_embeddings = self.model_sbert.encode(pmid_texts, convert_to_tensor=True, show_progress_bar=False)
            top_k = len(pmid_embeddings)
            
            if top_k == 0:
                print(f"No embeddings found for PMIDs in edge {edge}, skipping...")
                continue
            
            search_results = util.semantic_search(query_embedding, pmid_embeddings, top_k=top_k)
            
            if not search_results or not search_results[0]:
                print(f"No search results for edge {edge}, skipping...")
                continue
            
            best_pmid_index = search_results[0][0]['corpus_id']
            
            if best_pmid_index >= len(pmids):
                print(f"Invalid index {best_pmid_index} for PMIDs in edge {edge}, skipping...")
                continue
            
            best_pmid = pmids[best_pmid_index]
            best_score = search_results[0][0]['score']
            
            best_selection[edge] = best_pmid
            cosine_similarities[edge] = best_score
        
        return best_selection, cosine_similarities

    def generate_llm_response(self, best_selection, source, target):
        pmids = list(best_selection.values())
        path_context_abstracts = self.retrieve_abstracts(pmids)
        
        llm_prompt_template = "How would you describe an indirect relationship between {source} and {target} given the following scientific abstracts as contexts?"
        llm_fix_prompt_str = "\\n\\n".join(path_context_abstracts)
        llm_fix_prompt_combo = llm_prompt_template.format(source=source, target=target) + "\\n\\n" + llm_fix_prompt_str
        
        start_time = time.time()
        llm_resp_path = oai_get_response(llm_fix_prompt_combo, temp=1e-19, top_p=1e-9, seed=1234)
        end_time = time.time()
        
        print(f"Time taken for generating oai_get_response from llm: {end_time - start_time:.2f} seconds")
        
        return llm_resp_path

    def process_dataframe_row(self, row):
        source = "amiodarone"
        target = "mefloquine"
        edges_pmids = row['context_pmids']
        best_selection, cosine_similarities = self.find_best_selection(source, target, edges_pmids)
        return pd.Series({'best_selection_sbert': best_selection, 'cosine_similarity': cosine_similarities})

    def explain_path(self, source, target, edges_pmids):
        best_selection, cosine_similarities = self.find_best_selection(source, target, edges_pmids)
        llm_response = self.generate_llm_response(best_selection, source, target)
        return llm_response