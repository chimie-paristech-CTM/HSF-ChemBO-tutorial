import torch
import numpy as np
from typing import Type, Union, Literal, Optional, Any
import inspect
from pathlib import Path
from urllib.request import urlretrieve
from rdkit.Chem import MolFromSmiles, Mol
# For CheMeleon:
from chemprop import featurizers, nn
from chemprop.data import BatchMolGraph
from chemprop.nn import RegressionFFN
from chemprop.models import MPNN
# For ChemBERTa:
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModelForMaskedLM # pip install transformers
# For LLMs:
from base.llm_utils import get_model_and_tokenizer, average_pool, last_token_pool, weighted_average_pool
from functools import partial
from tqdm import tqdm
import torch.nn.functional as F

class CheMeleonFingerprint:
    def __init__(self, device: str | torch.device | None = None):
        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        agg = nn.MeanAggregation()
        ckpt_dir = Path().home() / ".chemprop"
        ckpt_dir.mkdir(exist_ok=True)
        mp_path = ckpt_dir / "chemeleon_mp.pt"
        if not mp_path.exists():
            urlretrieve(
                r"https://zenodo.org/records/15460715/files/chemeleon_mp.pt",
                mp_path,
            )
        chemeleon_mp = torch.load(mp_path, weights_only=True)
        mp = nn.BondMessagePassing(**chemeleon_mp['hyper_parameters'])
        mp.load_state_dict(chemeleon_mp['state_dict'])
        self.model = MPNN(
            message_passing=mp,
            agg=agg,
            predictor=RegressionFFN(input_dim=mp.output_dim),  # not actually used
        )
        self.model.eval()
        if device is not None:
            self.model.to(device=device)

    def __call__(self, molecules: list[str | Mol]) -> np.ndarray:
        bmg = BatchMolGraph([self.featurizer(MolFromSmiles(m) if isinstance(m, str) else m) for m in molecules])
        bmg.to(device=self.model.device)
        with torch.no_grad():
            return self.model.fingerprint(bmg).numpy(force=True)
        
        
class ChemBERTa_Fingerprint:
    '''
    model: seyonec/ChemBERTa-zinc-base-v1 \n
    type: torch.float32 \n
    size: 768
    
    model: DeepChem/ChemBERTa-100M-MLM \n
    type: torch.float32 \n
    size: 768
    '''
    def __init__(self, variant: Optional[Literal["zinc-base-v1","deepchem-100M-MLM"]] = None):
        if variant == "zinc-base-v1" or variant is None:
            self.model = AutoModelForMaskedLM.from_pretrained("seyonec/ChemBERTa-zinc-base-v1", cache_dir="./from_pretrained", local_files_only=False)
            self.tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1", cache_dir="./from_pretrained", local_files_only=False)
            # from_pretrained("seyonec/ChemBERTa-zinc-base-v1", cache_dir="./from_pretrained", local_files_only=True) # Download in advance and then load.
            
        elif variant == "deepchem-100M-MLM":
            self.tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-100M-MLM", cache_dir="./from_pretrained", local_files_only=False)
            self.model = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-100M-MLM", cache_dir="./from_pretrained", local_files_only=False)
            
        else:
            pass
    
    def __call__(self, smiles_list) -> np.ndarray:
        tokenized_smiles = [
            self.tokenizer(smile, return_tensors="pt") for smile in smiles_list
        ]
        outputs = [
            self.model(
                input_ids=tokenized_smile["input_ids"],
                attention_mask=tokenized_smile["attention_mask"],
                output_hidden_states=True,
            )
            for tokenized_smile in tokenized_smiles
        ]
        embeddings = torch.cat(
            [output["hidden_states"][0].sum(axis=1) for output in outputs], axis=0
        )
        return embeddings.detach().numpy()
    
class LLM_Fingerprint:
    '''
    model: t5-base \n
    size: 768
    
    model: WhereIsAI/UAE-Large-V1 \n
    size: 1024
    
    model: GT4SD/multitask-text-and-chemistry-t5-base-augm \n
    size: 768
    '''
    def __init__(self, model_name: Optional[Literal["t5-base","WhereIsAI/UAE-Large-V1", "GT4SD/multitask-text-and-chemistry-t5-base-augm"]] = 't5-base', pooling_method: Optional[Literal['average', 'cls', 'last_token_pool', 'weighted_average']] = 'average', normalize_embeddings = False, device='cpu', prefix=None):

        self.model_name = model_name
        self.pooling_method = pooling_method
        self.normalize_embeddings = normalize_embeddings
        self.device = device
        self.model, self.tokenizer = get_model_and_tokenizer(model_name, device)
        left_padding = self.tokenizer.padding_side == "left"
        self.model.eval()
        
        self.prefix = prefix
        
        self.pooling_functions = {
            "average": average_pool,
            "cls": lambda x, _: x[:, 0],
            "last_token_pool": partial(last_token_pool, left_padding=left_padding),
            "weighted_average": weighted_average_pool,
        }
    
    def __call__(self, texts) -> np.ndarray:
        # optionally add prefix to each text
        if self.prefix:
            texts = [self.prefix + text for text in texts]
        
        batch_size = 8
        max_length = 512
        
        embeddings_list = []
        for i in tqdm(
            range(0, len(texts), batch_size), desc=f"Processing with {self.model_name}", disable=True,
        ):
            batch_texts = texts[i : i + batch_size]
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoded_input)
                pooled = self.pooling_functions[self.pooling_method](
                    outputs.last_hidden_state, encoded_input["attention_mask"]
                )

                if self.normalize_embeddings:
                    pooled = F.normalize(pooled, p=2, dim=1)
                embeddings_list.append(pooled.cpu().numpy())

            torch.cuda.empty_cache()

        return np.concatenate(embeddings_list, axis=0)
    
    
class PretrainedWrapper:
    """
    Wrapper that ensures Pre-trained Model is always created and used in float32 mode.
    NOTE: BayBE uses float64
    
    NOTE: Could be problematic when multi-processing asynchronously.
    """
    def __init__(self, 
                pretrained_fingerprinter: Union[Type[CheMeleonFingerprint],Type[ChemBERTa_Fingerprint], Type[LLM_Fingerprint]], 
                **kwargs: Any,
                ):
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32) 
            
            # Filter correct __init__ parameters for fingerprint class. 
            sig = inspect.signature(pretrained_fingerprinter)
            valid_params = sig.parameters.keys()
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
            
            self._fingerprinter = pretrained_fingerprinter(**filtered_kwargs)
            
        finally:
            torch.set_default_dtype(original_dtype)
    
    def __call__(self, smiles_list):
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32)
            if isinstance(smiles_list, str):
                smiles_list = [smiles_list]
            return self._fingerprinter(smiles_list)
        finally:
            torch.set_default_dtype(original_dtype)
