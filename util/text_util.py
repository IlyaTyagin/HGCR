# def get_abstr_text(abstr_id, sents_db) -> str:
#     cur_text = ''
#     for i in range(100):
#         cur_sent_id = f's:{abstr_id}:1:{i}'
#         #print(cur_sent_id)
#         if cur_sent_id in sents_db:
#             cur_text += (
#               sents_db[cur_sent_id]['sent_text']
#             )
#         else:
#             break
#     return cur_text

def get_abstr_text(abstr_id, abstr_db) -> str:
    return abstr_db[abstr_id]

def get_path_context(
    edge_to_pmid_dict,
    sents_db,
) -> dict:
    
    """
    Given a dict of edges from a path and their PMIDs for every edge,
    returns texts for every PMID.
    """
    
    edge_to_pmid_text_dict = dict()
    
    for edge in edge_to_pmid_dict:
        pmid_list = edge_to_pmid_dict[edge]
        
        cur_pmid_dict = dict()
        for pmid in pmid_list:
            cur_pmid_dict[pmid] = get_abstr_text(pmid, sents_db)
        
        edge_to_pmid_text_dict[edge] = cur_pmid_dict
    
    return edge_to_pmid_text_dict