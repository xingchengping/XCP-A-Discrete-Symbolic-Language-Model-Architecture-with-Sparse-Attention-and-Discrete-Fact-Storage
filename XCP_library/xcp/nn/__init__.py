"""XCP's nn module (layers), mirroring the common subset of torch.nn."""
from .module import Module
from .embedding import DiscreteEmbedding
from .ffn import DiscreteFFN
from .attention import SparseAttention
from .relation import RelationGate
from .model import XCPModel

__all__ = ['Module', 'DiscreteEmbedding', 'DiscreteFFN', 'SparseAttention',
           'RelationGate', 'XCPModel']
