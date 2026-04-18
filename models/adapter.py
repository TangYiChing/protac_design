import torch
from torch_geometric.data import Data, Batch

class DiffusionToScorerAdapter:
    """Convert DiffPROTAC output to LinkerScorer input"""
    
    @staticmethod
    def extract_linker_batch(diffusion_output):
        """
        From PROTAC to linker
        
        Args:
            diffusion_output: DiffPROTAC"s sample_chain outputs
        Returns:
            PyG Batch object
        """
        positions = diffusion_output['positions']     # [B, N, 3]
        one_hot = diffusion_output['one_hot']         # [B, N, 9]
        linker_mask = diffusion_output['linker_mask'] # [B, N, 1]
        
        batch_size = positions.shape[0]
        data_list = []
        
        for i in range(batch_size):
            # get sample's linker mask
            mask = linker_mask[i].squeeze(-1).bool()  # [N]
            
            # keep linker atoms
            linker_pos = positions[i][mask]    # [N_linker, 3]
            linker_x = one_hot[i][mask]        # [N_linker, 9]
            
            # create PyG Data object
            data = Data(
                pos=linker_pos,
                x=linker_x,
                # edge_index determined by SAScorer
            )
            data_list.append(data)
        
        # merge into batch
        return Batch.from_data_list(data_list)
