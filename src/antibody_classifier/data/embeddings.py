"""
ESM-2 embedding extraction module.
Handles loading ESM-2 model and extracting sequence embeddings.
"""
import torch
import esm
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import setup_logger

logger = setup_logger(__name__)


class ESM2EmbeddingExtractor:
    """
    Extractor for ESM-2 protein language model embeddings.
    Handles batch processing and efficient CPU utilisation.
    """
    
    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        repr_layer: int = 33,
        batch_size: int = 8,
        use_mean_pooling: bool = True,
        device: Optional[str] = None,
        num_threads: Optional[int] = None,
        random_seed: int = 42
    ):
        """
        Initialise ESM-2 embedding extractor.
        
        Args:
            model_name: ESM-2 model variant name
            repr_layer: Layer to extract representations from
            batch_size: Batch size for processing
            use_mean_pooling: Whether to mean-pool across sequence length
            device: Device to use ('cpu' or 'cuda'). Auto-detect if None
            num_threads: Number of CPU threads to use. Auto-detect if None
            random_seed: Random seed for reproducibility
        """
        self.model_name = model_name
        self.repr_layer = repr_layer
        self.batch_size = batch_size
        self.use_mean_pooling = use_mean_pooling
        
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        if self.device.type == 'cpu':
            if num_threads is not None:
                torch.set_num_threads(num_threads)
            actual_threads = torch.get_num_threads()
            logger.info(f"PyTorch CPU threads: {actual_threads}")
        
        logger.info(f"Initialising ESM2EmbeddingExtractor")
        logger.info(f"Model: {model_name}")
        logger.info(f"Representation layer: {repr_layer}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mean pooling: {use_mean_pooling}")
        logger.info(f"Random seed: {random_seed}")
        
        self._load_model()
    
    def _load_model(self):
        """Load ESM-2 model and alphabet."""
        logger.info("Loading ESM-2 model...")
        
        try:
            self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(
                self.model_name
            )
            
            self.model = self.model.to(self.device)
            self.model.eval()  
            self.batch_converter = self.alphabet.get_batch_converter()
            logger.info(f"Model loaded successfully")
            logger.info(f"Model size: {sum(p.numel() for p in self.model.parameters()):,} parameters")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def extract_single_embedding(self, sequence: str, label: str = "seq") -> np.ndarray:
        """
        Extract embedding for a single sequence.
        
        Args:
            sequence: Protein sequence string
            label: Sequence label/identifier
        
        Returns:
            Embedding array
        """
        data = [(label, sequence)]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)
        
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[self.repr_layer])
            token_representations = results["representations"][self.repr_layer]
        
        if self.use_mean_pooling:
            sequence_length = len(sequence)
            embedding = token_representations[0, 1:sequence_length+1].mean(0)
        else:
            embedding = token_representations[0, 0]
        
        return embedding.cpu().numpy()
    
    def extract_batch_embeddings(
        self,
        sequences: List[str],
        labels: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Extract embeddings for a batch of sequences.
        
        Args:
            sequences: List of protein sequences
            labels: Optional list of sequence labels
        
        Returns:
            Array of embeddings (batch_size, embedding_dim)
        """
        if labels is None:
            labels = [f"seq_{i}" for i in range(len(sequences))]
        
        data = list(zip(labels, sequences))
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)
        
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[self.repr_layer])
            token_representations = results["representations"][self.repr_layer]
        
        embeddings = []
        for i, seq in enumerate(sequences):
            if self.use_mean_pooling:
                sequence_length = len(seq)
                embedding = token_representations[i, 1:sequence_length+1].mean(0)
            else:
                embedding = token_representations[i, 0]
            
            embeddings.append(embedding.cpu().numpy())
        
        return np.array(embeddings)
    
    def extract_embeddings_from_dataframe(
        self,
        df: pd.DataFrame,
        sequence_column: str = "Sequence",
        id_column: str = "POI"
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract embeddings for all sequences in a dataframe.
        
        Args:
            df: Dataframe containing sequences
            sequence_column: Name of column containing sequences
            id_column: Name of column containing sequence IDs
        
        Returns:
            Tuple of (embeddings array, list of IDs)
        """
        logger.info(f"Extracting embeddings for {len(df):,} sequences")
        logger.info(f"Batch size: {self.batch_size}")
        
        sequences = df[sequence_column].tolist()
        ids = df[id_column].tolist()
        
        embeddings = []
        n_batches = (len(sequences) + self.batch_size - 1) // self.batch_size
        
        pbar = tqdm(total=len(sequences), desc="Extracting embeddings")
        
        for i in range(0, len(sequences), self.batch_size):
            batch_sequences = sequences[i:i + self.batch_size]
            batch_ids = ids[i:i + self.batch_size]
            
            try:
                batch_embeddings = self.extract_batch_embeddings(
                    batch_sequences,
                    batch_ids
                )
                embeddings.append(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Error processing batch {i//self.batch_size + 1}/{n_batches}: {e}")
                logger.info("Falling back to single-sequence processing for this batch")
                for seq, seq_id in zip(batch_sequences, batch_ids):
                    try:
                        emb = self.extract_single_embedding(seq, seq_id)
                        embeddings.append(emb[np.newaxis, :])
                    except Exception as e2:
                        logger.error(f"Failed to process sequence {seq_id}: {e2}")
                        emb_dim = embeddings[0].shape[-1] if embeddings else 1280
                        embeddings.append(np.zeros((1, emb_dim)))
            
            pbar.update(len(batch_sequences))
        
        pbar.close()
        
        embeddings = np.concatenate(embeddings, axis=0)
        
        logger.info(f"Extracted embeddings shape: {embeddings.shape}")
        
        return embeddings, ids
    
    def save_embeddings(
        self,
        embeddings: np.ndarray,
        ids: List[str],
        labels: np.ndarray,
        output_path: str,
        metadata: Optional[Dict] = None
    ):
        """
        Save embeddings to disk.
        
        Args:
            embeddings: Embeddings array
            ids: List of sequence IDs
            output_path: Path to save embeddings (.npz format)
            metadata: Optional metadata dictionary
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving embeddings to {output_path}")
        
        save_dict = {
            'embeddings': embeddings,
            'ids': np.array(ids),
            'labels': np.array(labels)
        }
        
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'model_name': self.model_name,
            'repr_layer': self.repr_layer,
            'use_mean_pooling': self.use_mean_pooling,
            'embedding_dim': embeddings.shape[1],
            'n_sequences': len(ids)
        })
        
        for key, value in metadata.items():
            save_dict[f'metadata_{key}'] = str(value)
        
        np.savez_compressed(output_file, **save_dict)
        
        logger.info(f"Saved embeddings: {embeddings.shape}")
        logger.info(f"File size: {output_file.stat().st_size / (1024**2):.2f} MB")
    
    def load_embeddings(self, input_path: str) -> Tuple[np.ndarray, List[str], Dict]:
        """
        Load embeddings from disk.
        
        Args:
            input_path: Path to embeddings file (.npz format)
        
        Returns:
            Tuple of (embeddings array, list of IDs, metadata dict)
        """
        logger.info(f"Loading embeddings from {input_path}")
        
        data = np.load(input_path, allow_pickle=True)
        embeddings = data['embeddings']
        ids = data['ids'].tolist()
        metadata = {}
        for key in data.files:
            if key.startswith('metadata_'):
                metadata_key = key.replace('metadata_', '')
                metadata[metadata_key] = str(data[key])
        
        logger.info(f"Loaded embeddings shape: {embeddings.shape}")
        logger.info(f"Metadata: {metadata}")
        
        return embeddings, ids, metadata


def extract_and_save_embeddings(
    data_splits: Dict[str, pd.DataFrame],
    output_dir: str,
    model_name: str = "esm2_t33_650M_UR50D",
    batch_size: int = 8,
    num_threads: Optional[int] = None,
    random_seed: int = 42,
    sequence_column: str = "Sequence"
) -> Dict[str, str]:
    """
    Extract and save embeddings for all data splits.
    
    Args:
        data_splits: Dictionary with 'train', 'val', 'test' dataframes
        output_dir: Directory to save embeddings
        model_name: ESM-2 model name
        batch_size: Batch size for processing
        num_threads: Number of CPU threads to use
        random_seed: Random seed for reproducibility
        sequence_column: Name of sequence column
    
    Returns:
        Dictionary mapping split names to embedding file paths
    """
    logger.info("EXTRACTING ESM-2 EMBEDDINGS")
    
    extractor = ESM2EmbeddingExtractor(
        model_name=model_name,
        batch_size=batch_size,
        num_threads=num_threads,
        random_seed=random_seed
    )
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    embedding_paths = {}
    
    # Process each split
    for split_name, df in data_splits.items():
        logger.info(f"\nProcessing {split_name} split ({len(df):,} sequences)")
        embeddings, ids = extractor.extract_embeddings_from_dataframe(
            df,
            sequence_column=sequence_column
        )
        labels = df['label'].values
        output_file = output_path / f"{split_name}_embeddings.npz"
        extractor.save_embeddings(
            embeddings,
            ids,
            labels,
            str(output_file),
            metadata={'split': split_name}
        )
        embedding_paths[split_name] = str(output_file)
    
    logger.info("EMBEDDING EXTRACTION COMPLETE")
    
    return embedding_paths