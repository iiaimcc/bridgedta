import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv


def batch_nodes_to_padded(node_features, batch):
    """
    Convert PyG batch graph node features to a padded 3D tensor.
    Args:
        node_features: [total_nodes, feat_dim]
        batch: [total_nodes] graph index each node belongs to
    Returns:
        padded: [batch_size, max_nodes, feat_dim]
        mask: [batch_size, max_nodes] (1=valid, 0=padding)
    """
    batch_size = batch.max().item() + 1
    feat_dim = node_features.size(-1)

    # Count nodes per graph
    counts = torch.zeros(batch_size, dtype=torch.long, device=batch.device)
    counts.scatter_add_(0, batch, torch.ones_like(batch, dtype=torch.long))
    max_nodes = counts.max().item()

    padded = torch.zeros(batch_size, max_nodes, feat_dim, device=node_features.device)
    mask = torch.zeros(batch_size, max_nodes, device=node_features.device)

    # Compute local index of each node within its graph
    offsets = torch.zeros_like(batch)
    for i in range(batch_size):
        graph_mask = (batch == i)
        num_nodes_i = graph_mask.sum().item()
        offsets[graph_mask] = torch.arange(num_nodes_i, device=batch.device)
        padded[i, :num_nodes_i] = node_features[graph_mask]
        mask[i, :num_nodes_i] = 1.0

    return padded, mask


class DrugGNN(nn.Module):
    """
    Lightweight Drug Expert: 3-layer GINConv + residual connections
    """
    def __init__(self, num_feat_in=78, num_feat_out=256, hidden_dim=128, dr=0.2):
        super(DrugGNN, self).__init__()
        self.num_feat_in = num_feat_in
        self.num_feat_out = num_feat_out
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dr)
        self.relu = nn.ReLU()

        # Input projection: map raw atom features to hidden_dim for residual alignment
        self.input_proj = nn.Linear(num_feat_in, hidden_dim)

        # 3-layer GINConv + BatchNorm, each MLP has 2 layers to maintain expressiveness
        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ))
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ))
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.conv3 = GINConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ))
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # Node-level output projection: hidden_dim -> num_feat_out
        self.fc_out = nn.Linear(hidden_dim, num_feat_out)

    def forward(self, xd, edge_index, batch):
        """
        Returns:
            node_features: atom-level features [total_nodes, num_feat_out]
            batch: batch index of each node (pass-through)
        """
        # Input projection
        h = self.relu(self.input_proj(xd))

        # 3-layer GINConv + residual connections
        h1 = self.bn1(self.relu(self.conv1(h, edge_index)))
        h1 = h1 + h

        h2 = self.bn2(self.relu(self.conv2(h1, edge_index)))
        h2 = h2 + h1

        h3 = self.bn3(self.relu(self.conv3(h2, edge_index)))
        h3 = h3 + h2

        # Node-level output (preserve atom dimension for attention visualization)
        node_out = self.dropout(self.relu(self.fc_out(h3)))
        return node_out, batch

class FASTAModel(nn.Module):
    """
    Lightweight Protein Expert: multi-scale CNN + residual connections + BatchNorm
    """
    def __init__(self, char_set_len, num_feat_out=256, embed_dim=128, hidden_dim=128, dr=0.1):
        super().__init__()
        self.num_feat_out = num_feat_out
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(char_set_len, embed_dim)

        # Input projection: map embedding to hidden_dim for residual alignment
        self.input_proj = nn.Conv1d(embed_dim, hidden_dim, kernel_size=1)
        self.in_bn = nn.BatchNorm1d(hidden_dim)

        # Multi-scale convolution blocks (k=4/8/12), padding='same' preserves sequence length for residual addition
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, padding='same')
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=8, padding='same')
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=12, padding='same')
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # Output projection after Global MaxPool: map hidden_dim to num_feat_out
        self.fc_out = nn.Linear(hidden_dim, num_feat_out)
        self.dropout = nn.Dropout(dr)
        self.relu = nn.ReLU()

    def forward(self, fasta):
        # keras is channel last, different with pytorch
        # [B, L] -> [B, embed_dim, L]
        v = self.embed(fasta).transpose(1, 2)

        # Input projection + BN, serving as baseline for subsequent residuals
        h = self.relu(self.in_bn(self.input_proj(v)))

        # Three-layer multi-scale convolution + residual
        h1 = self.relu(self.bn1(self.conv1(h)))
        h1 = h1 + h

        h2 = self.relu(self.bn2(self.conv2(h1)))
        h2 = h2 + h1

        h3 = self.relu(self.bn3(self.conv3(h2)))
        h3 = h3 + h2

        # Final Global MaxPool: [B, hidden_dim]
        v, _ = torch.max(h3, -1)

        # Output projection to num_feat_out=256, aligned with DrugGNN
        out = self.dropout(self.relu(self.fc_out(v)))
        return out

class BridgeModule(nn.Module):
    """
    Cross-attention-based bridge module for fusing LPM features and Expert features
    """
    def __init__(self, expert_dim, lpm_dim, num_heads=8, dropout=0.1):
        super(BridgeModule, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = expert_dim  # hidden_dim equals expert_dim, Expert is not projected
        self.head_dim = self.hidden_dim // num_heads

        # Only project LPM features to Expert dimension space
        self.lpm_projection = nn.Linear(lpm_dim, self.hidden_dim)

        # Multi-head attention Q/K/V projections
        self.query_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        # LayerNorm and Dropout
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # [batch, heads, seq_len, head_dim]

    def combine_heads(self, x):
        batch_size, _, seq_len, _ = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, seq_len, self.hidden_dim)

    def forward(self, expert_features, lpm_features, mask=None, return_attn=False):
        """
        Args:
            expert_features: Expert encoder output [batch, seq_q, expert_dim]
            lpm_features: Frozen LPM encoder output [batch, (seq_k,) lpm_dim]
            mask: Optional attention mask [batch, seq_k]
            return_attn: Whether to return attention weights for visualization
        Returns:
            output: Fused features [batch, seq_q, hidden_dim]
            attn_weights: (Optional) attention weights [batch, num_heads, seq_q, seq_k]
        """
        # Ensure LPM input is 3D: [batch, seq_k, lpm_dim]
        if len(lpm_features.shape) == 2:
            lpm_features = lpm_features.unsqueeze(1)

        # Ensure Expert input is 3D: [batch, seq_q, expert_dim]
        if len(expert_features.shape) == 2:
            expert_features = expert_features.unsqueeze(1)

        # Only project LPM to Expert dimension space; Expert serves directly as Q
        lpm_proj = self.lpm_projection(lpm_features)  # [batch, seq_k, hidden_dim]

        # Expert as Query, projected LPM as Key and Value
        query = self.split_heads(self.query_proj(expert_features))
        key = self.split_heads(self.key_proj(lpm_proj))
        value = self.split_heads(self.value_proj(lpm_proj))

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_k]
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)

        attn_weights = F.softmax(scores, dim=-1)
        attn_out = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_out, value)

        # Combine multi-head outputs and apply output projection
        attn_output = self.combine_heads(attn_output)
        attn_output = self.output_proj(attn_output)

        # Residual connection: add back Expert features, then LayerNorm
        output = self.layer_norm(attn_output + expert_features)

        if return_attn:
            return output, attn_weights
        return output


class InteractionCrossAttention(nn.Module):
    """
    Bidirectional cross-attention module for drug-protein interaction fusion
    """
    def __init__(self, drug_dim, protein_dim, hidden_dim=128, num_heads=8, dropout=0.1, ffn_ratio=2):
        super(InteractionCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        # Projection layers
        self.drug_projection = nn.Linear(drug_dim, hidden_dim)
        self.protein_projection = nn.Linear(protein_dim, hidden_dim)

        # Drug attends to protein Q/K/V
        self.drug_query = nn.Linear(hidden_dim, hidden_dim)
        self.prot_key_for_drug = nn.Linear(hidden_dim, hidden_dim)
        self.prot_value_for_drug = nn.Linear(hidden_dim, hidden_dim)

        # Protein attends to drug Q/K/V
        self.prot_query = nn.Linear(hidden_dim, hidden_dim)
        self.drug_key_for_prot = nn.Linear(hidden_dim, hidden_dim)
        self.drug_value_for_prot = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.drug_output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.prot_output_proj = nn.Linear(hidden_dim, hidden_dim)

        # Attention residual + LayerNorm after attention sub-layer
        self.drug_attn_norm = nn.LayerNorm(hidden_dim)
        self.prot_attn_norm = nn.LayerNorm(hidden_dim)

        # FFN sub-layer + LayerNorm (residual -> LN)
        ffn_dim = hidden_dim * ffn_ratio
        self.drug_ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.prot_ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.drug_ffn_norm = nn.LayerNorm(hidden_dim)
        self.prot_ffn_norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def combine_heads(self, x):
        batch_size, _, seq_len, _ = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, seq_len, self.hidden_dim)

    def forward(self, drug_features, protein_features, drug_mask=None):
        """
        Args:
            drug_features: Fused drug features [batch, seq_d, drug_dim]
            protein_features: Fused protein features [batch, seq_p, protein_dim]
            drug_mask: Drug atom padding mask [batch, seq_d], 1=real atom, 0=padding
        Returns:
            Concatenated interaction features [batch, hidden_dim * 2]
        """
        if len(drug_features.shape) == 2:
            drug_features = drug_features.unsqueeze(1)
        if len(protein_features.shape) == 2:
            protein_features = protein_features.unsqueeze(1)

        drug_proj = self.drug_projection(drug_features)
        prot_proj = self.protein_projection(protein_features)

        # Drug attends to protein
        q_drug = self.split_heads(self.drug_query(drug_proj))
        k_prot = self.split_heads(self.prot_key_for_drug(prot_proj))
        v_prot = self.split_heads(self.prot_value_for_drug(prot_proj))

        scores_d2p = torch.matmul(q_drug, k_prot.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_d2p = F.softmax(scores_d2p, dim=-1)
        attn_d2p = self.dropout(attn_d2p)
        drug_attn_out = self.combine_heads(torch.matmul(attn_d2p, v_prot))
        drug_attn_out = self.drug_output_proj(drug_attn_out)

        # Attention residual + LN
        drug_h = self.drug_attn_norm(drug_proj + self.dropout(drug_attn_out))
        # FFN residual + LN
        drug_h = self.drug_ffn_norm(drug_h + self.dropout(self.drug_ffn(drug_h)))

        # Protein attends to drug
        q_prot = self.split_heads(self.prot_query(prot_proj))
        k_drug = self.split_heads(self.drug_key_for_prot(drug_proj))
        v_drug = self.split_heads(self.drug_value_for_prot(drug_proj))

        scores_p2d = torch.matmul(q_prot, k_drug.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # Mask padding atoms in K dimension to prevent protein from attending to fake atom positions
        if drug_mask is not None:
            kv_mask = drug_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_d]
            scores_p2d = scores_p2d.masked_fill(kv_mask == 0, torch.finfo(scores_p2d.dtype).min)
        attn_p2d = F.softmax(scores_p2d, dim=-1)
        attn_p2d = self.dropout(attn_p2d)
        prot_attn_out = self.combine_heads(torch.matmul(attn_p2d, v_drug))
        prot_attn_out = self.prot_output_proj(prot_attn_out)

        # Attention residual + LN
        prot_h = self.prot_attn_norm(prot_proj + self.dropout(prot_attn_out))
        # FFN residual + LN
        prot_h = self.prot_ffn_norm(prot_h + self.dropout(self.prot_ffn(prot_h)))

        # Pool and concatenate: use mask-weighted average for drug to avoid padding atoms diluting real atom info
        if drug_mask is not None:
            mask_f = drug_mask.unsqueeze(-1).to(drug_h.dtype)        # [batch, seq_d, 1]
            denom = mask_f.sum(dim=1).clamp(min=1.0)                  # [batch, 1]
            drug_out = (drug_h * mask_f).sum(dim=1) / denom           # [batch, hidden_dim]
        else:
            drug_out = drug_h.mean(dim=1)
        prot_out = prot_h.mean(dim=1)    # [batch, hidden_dim]

        fused_features = torch.cat([drug_out, prot_out], dim=1)
        return fused_features

class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024):
        super(Predictor, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        return self.fc_layers(x).squeeze(-1)

class CILoss(nn.Module):
    """Concordance Index (CI) Loss for ranking prediction values within a batch"""
    def __init__(self, margin=0.1):
        super(CILoss, self).__init__()
        self.margin = margin
        
    def forward(self, pred, true_values):
        """
        Args:
            pred: Predicted values [batch_size]
            true_values: Ground truth values [batch_size]
        Returns:
            CI loss value
        """
        # Create pairwise GT mask where true_i > true_j
        batch_size = true_values.size(0)
        if batch_size <= 1:  # Cannot compute CI loss with single example
            return torch.tensor(0.0, device=pred.device)
            
        # Create all possible pairs within the batch
        true_matrix = true_values.view(-1, 1) - true_values.view(1, -1)
        gt_mask = (true_matrix > 0).float()
        
        # Create prediction difference matrix
        pred_matrix = pred.view(-1, 1) - pred.view(1, -1)
        
        # Compute the CI loss using a margin-based approach
        # For each pair where true_i > true_j, we want pred_i > pred_j
        ranking_loss = F.relu(self.margin - pred_matrix)
        
        # Only consider pairs where true_i > true_j
        masked_ranking_loss = ranking_loss * gt_mask
        
        # Normalize by the number of valid pairs
        valid_pairs = torch.sum(gt_mask)
        if valid_pairs > 0:
            return torch.sum(masked_ranking_loss) / valid_pairs
        else:
            return torch.tensor(0.0, device=pred.device)

class BridgeDTA(nn.Module):
    def __init__(self, fasta_char_set_len,
                 num_feat_xd=78, num_feat_out=128, dr=0.2,
                 kpgt_dim=768):
        super(BridgeDTA, self).__init__()

        self.drug_gnn_dim = 256      # GNN Expert output dimension
        self.kpgt_dim = kpgt_dim     # KPGT LPM output dimension
        self.protein_cnn_dim = 256   # CNN Expert output dimension
        self.protein_esm_dim = 640   # ESM-2 LPM output dimension
        
        # Drug graph model (Expert)
        self.drug_gnn = DrugGNN(
            num_feat_in=num_feat_xd,
            num_feat_out=self.drug_gnn_dim,
            hidden_dim=128,
            dr=dr,
        )
        
        # Protein sequence model (Expert)
        self.fasta_model = FASTAModel(
            fasta_char_set_len,
            num_feat_out=self.protein_cnn_dim,
            embed_dim=128,
            hidden_dim=128,
            dr=dr,
        )
        
        # Drug feature fusion
        self.drug_bridge = BridgeModule(
            expert_dim=self.drug_gnn_dim,
            lpm_dim=self.kpgt_dim,
            num_heads=8
        )
        
        # Protein feature fusion
        self.protein_bridge = BridgeModule(
            expert_dim=self.protein_cnn_dim,
            lpm_dim=self.protein_esm_dim,
            num_heads=8
        )
        
        # Drug-protein interaction fusion
        self.interaction_cross_attn = InteractionCrossAttention(
            drug_dim=self.drug_gnn_dim,
            protein_dim=self.protein_cnn_dim,
            hidden_dim=self.drug_gnn_dim,
            num_heads=8
        )
        
        # Final predictor
        self.predictor = Predictor(self.drug_gnn_dim * 2)
        
        # CI Loss for auxiliary task
        self.ci_loss = CILoss(margin=0.1)
        
    def forward(self, data, fasta, kpgt_features, esm_features, esm_mask,
                kpgt_mask=None, true_values=None, return_attn=False):
        """
        Args:
            data: PyG Batch object (drug molecular graph)
            fasta: Protein sequence encoding [batch, seq_len]
            kpgt_features: KPGT frozen features
                - graph-level: [batch, kpgt_dim] or [batch, 1, kpgt_dim]
                - token-level: [batch, max_atoms, kpgt_dim]
            esm_features: ESM-2 frozen features [batch, seq_len, esm_dim]
            esm_mask: ESM-2 attention mask [batch, seq_len]
            kpgt_mask: KPGT padding mask [batch, max_atoms], required for token-level, None for graph-level
            true_values: Ground truth affinity values (for CI-Loss)
            return_attn: Whether to return drug bridge attention weights for visualization
        Returns:
            output: Affinity prediction values [batch]
            ci_loss: CI-Loss value
            drug_attn_weights: (Optional) drug bridge attention [batch, heads, num_atoms, kpgt_seq]
        """
        # 1. Expert encoding: GNN extracts atom-level local structure features
        xd, edge_index, graph_batch = data.x, data.edge_index, data.batch
        drug_node_feat, _ = self.drug_gnn(xd, edge_index, graph_batch)

        # Convert PyG flat node features to padded 3D tensor [batch, max_atoms, drug_gnn_dim]
        drug_atom_feat, drug_mask = batch_nodes_to_padded(drug_node_feat, graph_batch)

        # 2. Expert encoding: CNN extracts protein local sequence features [batch, protein_cnn_dim]
        protein_cnn_feat = self.fasta_model(fasta)

        # 3. Drug feature fusion: atom-level Expert(Q) attends to KPGT LPM(K/V)
        if return_attn:
            drug_fused, drug_attn_weights = self.drug_bridge(
                drug_atom_feat, kpgt_features, mask=kpgt_mask, return_attn=True)
        else:
            drug_fused = self.drug_bridge(drug_atom_feat, kpgt_features, mask=kpgt_mask)
            drug_attn_weights = None
        # Mask padding atom positions in bridge output using drug_mask
        drug_fused = drug_fused * drug_mask.unsqueeze(-1)

        # 3.5 Graph-level pooling after Drug Bridge: atom-level -> graph-level
        mask_float = drug_mask.unsqueeze(-1)                          # [batch, max_atoms, 1]
        drug_pooled = (drug_fused * mask_float).sum(dim=1, keepdim=True) \
                      / mask_float.sum(dim=1, keepdim=True).clamp(min=1.0)  # [batch, 1, drug_gnn_dim]

        # 4. Protein feature fusion: CNN Expert(Q) attends to ESM-2 LPM(K/V)
        protein_fused = self.protein_bridge(protein_cnn_feat, esm_features, mask=esm_mask)

        # 5. Drug-protein interaction fusion (bidirectional cross-attention)
        #    drug_pooled: [batch, 1, drug_gnn_dim]
        #    protein_fused: [batch, 1, protein_cnn_dim]
        fused_features = self.interaction_cross_attn(drug_pooled, protein_fused)
        
        # 6. Predictor head outputs affinity score
        output = self.predictor(fused_features)
        
        # 7. Auxiliary ranking task: CI-Loss
        ci_loss = torch.tensor(0.0, device=output.device)
        if true_values is not None:
            ci_loss = self.ci_loss(output, true_values)
        
        if return_attn:
            return output, ci_loss, drug_attn_weights
        return output, ci_loss
    
    def compute_ci(self, pred, true_values):
        """
        Compute Concordance Index (CI) for evaluation
        Args:
            pred: Predicted values
            true_values: Ground truth values
        Returns:
            CI value (float)
        """
        with torch.no_grad():
            batch_size = true_values.size(0)
            if batch_size <= 1:
                return 1.0
                
            # Create all pairs
            true_matrix = true_values.view(-1, 1) > true_values.view(1, -1)
            pred_diff = pred.view(-1, 1) - pred.view(1, -1)
            
            # Calculate h_one (pred_i > pred_j) and h_half (pred_i = pred_j)
            h_one = (pred_diff > 0).float()
            h_half = (pred_diff == 0).float() * 0.5
            
            # Calculate CI as in metrics.py
            num_pairs = torch.sum(true_matrix.float())
            if num_pairs > 0:
                ci_value = torch.sum(true_matrix.float() * (h_one + h_half)) / num_pairs
                return ci_value.item()
            else:
                return 1.0  # No valid pairs