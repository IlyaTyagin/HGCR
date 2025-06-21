import numpy as np
import json
from joblib import Parallel, delayed
from pathlib import Path
from tqdm import tqdm

class MedCPTNumpyEmbeddings:
    def __init__(
        self,
        medcpt_fpath,
        read_n_jobs=1,
        memmap=True
    ):
        
        self.memmap = None
        if memmap:
            self.memmap = 'r'
        else:
            print('Memmap is not used, MedCPT embeddings are loaded to RAM.')
        
        self.medcpt_fpath = Path(medcpt_fpath)
        
        self.pmids_chunk_list = None
        self.emb_chunks_dict = None
        self.pmids_to_loc_dict = None
        
        self.emb_chunk_np_flist = sorted(self.medcpt_fpath.glob('embeds_chunk_*.npy'))
        self.pmids_chunk_json_flist = sorted(self.medcpt_fpath.glob('pmids_chunk*.json'))

        assert len(self.pmids_chunk_json_flist) == len(self.emb_chunk_np_flist)
        
        self.open_np_chunks(n_jobs=read_n_jobs)
        self.open_json_chunks(n_jobs=read_n_jobs)
        tqdm._instances.clear()
        self.construct_lookup_dict()
        
        return None
    
    def open_single_np_chunk(self, fname):
        
        k = fname.stem.split('_chunk_')[-1]
        v = np.load(
            fname,
            mmap_mode=self.memmap,
        )
        
        return (k,v)
    
    def open_np_chunks(self, n_jobs):
        
        #emb_chunks_dict = dict()

        # for fname in tqdm(
        #     self.emb_chunk_np_flist,
        #     desc='Opening np chunks'
        # ):
        #     k = fname.stem.split('_')[-1]
        #     v = np.load(
        #         fname,
        #         mmap_mode=self.memmap,
        #     )
        #     emb_chunks_dict[k] = v
            
        emb_chunks_np_list = (
            Parallel(n_jobs=n_jobs)(
                delayed(self.open_single_np_chunk)(fname) for fname in tqdm(
                    self.emb_chunk_np_flist,
                    desc='Opening np chunks'
                )
            )
        )
        
        #for k,v in pmids_chunk_np_list:
            
        emb_chunks_dict = dict(emb_chunks_np_list)
        
        self.emb_chunks_dict = emb_chunks_dict
        return None
    
    def open_json(self, fname):
        try:
            with open(fname, 'r') as f:
                js = json.load(f)
        except Exception as e:
            print(e)
            print(fname)
            return fname

        return js
    
    def open_json_chunks(self, n_jobs):
        self.pmids_chunk_list = (
            Parallel(n_jobs=n_jobs)(
                delayed(self.open_json)(fname) for fname in tqdm(
                    self.pmids_chunk_json_flist,
                    desc='Opening json index chunks'
                )
            )
        )
    
    def construct_lookup_dict(self):
        
        self.chunk_to_pmids_list_dict = dict(
            zip(
                [fname.stem.split('_chunk_')[-1] for fname in self.pmids_chunk_json_flist],
                self.pmids_chunk_list
            )
        )
        
        pmids_to_loc_dict = dict()

        for chunk_idx, pmids_list in tqdm(
            self.chunk_to_pmids_list_dict.items(),
            desc='Constructing PMID lookup index'
        ):
            for row_idx, pmid in enumerate(pmids_list):
                emb_loc = f'{chunk_idx}|{row_idx}'

                pmids_to_loc_dict[pmid] = emb_loc
                
        self.pmids_to_loc_dict = pmids_to_loc_dict
        
        return None
    
    def __getitem__(self, pmid):
        
        emb_loc = self.pmids_to_loc_dict[pmid]
    
        chunk_idx, row_idx = emb_loc.split('|')

        emb_np = self.emb_chunks_dict[chunk_idx][int(row_idx)]

        return emb_np

    def __len__(self):
        return len(self.pmids_to_loc_dict)
    
    def __contains__(self, el):
        return el in self.pmids_to_loc_dict