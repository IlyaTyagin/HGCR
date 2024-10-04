import json
from joblib import Parallel, delayed
from tqdm import tqdm

class MultiFileJsonlReader:
    def __init__(
        self,
        filenames,
        ignore_last_line=False,
        n_idx_jobs=20,
        extract_keys=True,
    ):
        self.filenames = filenames
        self.ignore_last_line = ignore_last_line
        self.n_idx_jobs = n_idx_jobs
        self.index, self.key_index = self._create_index(extract_keys=extract_keys)

    def _create_index(self, extract_keys=False):
        """Create an index for the starting byte of each line across multiple files using parallel processing."""
        def index_file(file_idx, filename, extract_keys):
            with open(filename, 'r') as file:
                local_index = []
                local_key_index = {}
                offset = file.tell()

                while True:
                    line = file.readline()
                    if not line:
                        break

                    local_index.append((file_idx, offset))
                    if extract_keys:
                        try:
                            data = json.loads(line)
                            key = data.get('key')
                            if key is not None:
                                local_key_index[key] = len(local_index) - 1  # Store line number relative to file index
                        except json.JSONDecodeError:
                            pass  # Handle the case where a line might be malformed

                    offset = file.tell()

                if self.ignore_last_line:
                    local_index = local_index[:-1]

                    # Also adjust the key index to remove last entry if ignore_last_line is True
                    if extract_keys:
                        if local_index:
                            last_line_number = len(local_index)  # Length gives the number of valid lines
                            keys_to_remove = [k for k, v in local_key_index.items() if v >= last_line_number]
                            for k in keys_to_remove:
                                del local_key_index[k]

                return local_index, local_key_index

        results = Parallel(n_jobs=self.n_idx_jobs, batch_size=1)(
            delayed(index_file)(i, filename, extract_keys) for i, filename in enumerate(
                tqdm(
                    self.filenames,
                    desc='Indexing jsonl files'
                )
            )
        )

        # Separate file indices and key indices from results
        combined_index = []
        combined_key_index = {}
        for local_index, local_key_index in results:
            # Adjust the line numbers in local_key_index to be absolute line numbers across all files
            current_offset = len(combined_index)
            for key, local_line_number in local_key_index.items():
                combined_key_index[key] = current_offset + local_line_number

            combined_index.extend(local_index)

        return combined_index, combined_key_index

    def __getitem__(self, line_number):
        """Retrieve a specific line from the appropriate file and parse it as JSON."""
        if line_number >= len(self.index) or line_number < 0:
            raise IndexError('Line number out of range')
        file_idx, offset = self.index[line_number]
        with open(self.filenames[file_idx], 'r') as file:
            file.seek(offset)
            line = file.readline()
            return json.loads(line)

    def get_by_key(self, key_obj):
        """Retrieve a specific line using the key_obj value."""
        if key_obj not in self.key_index:
            #raise KeyError(f'Key {key_obj} not found')
            return None
        line_number = self.key_index[key_obj]
        return self[line_number]

    def __len__(self):
        """Return the total number of lines across all files."""
        return len(self.index)

class LazyJsonlAbstractLoader(MultiFileJsonlReader):
    def __init__(
        self,
        filenames,
        n_idx_jobs=20,
    ):
        self.filenames = filenames
        self.ignore_last_line = False
        self.n_idx_jobs = n_idx_jobs
        self.index, self.key_index = self._create_index(extract_keys=True)
        
        return None
    
    def __getitem__(self, key_obj):
        """Retrieve a specific line using the key_obj value."""
        if key_obj not in self.key_index:
            #raise KeyError(f'Key {key_obj} not found')
            return None
        line_number = self.key_index[key_obj]
        kv_dict = super().__getitem__(line_number)

        abstr_text = ' '.join(
            [
                pair[1] for pair in (
                    sorted(
                        kv_dict['value'],
                        key=lambda x: int(x[0].split(':')[-1])
                    )
                )
            ]
        )
        return abstr_text
        
    