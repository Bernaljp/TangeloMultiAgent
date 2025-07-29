"""
Node2Vec graph embedding for preprocessing high-dimensional inputs.

This module implements Node2Vec embeddings to create dense representations
of graph nodes, reducing dimensionality before graph neural network processing.
"""

import warnings
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx
from gensim.models import Word2Vec

from ..config import GraphConfig


class Node2VecEmbedding:
    """
    Node2Vec embedding generator for graph preprocessing.
    
    Implements the Node2Vec algorithm to generate dense node embeddings
    from graph structure using biased random walks.
    
    Parameters
    ----------
    config : GraphConfig
        Configuration object containing Node2Vec parameters.
    """
    
    def __init__(self, config: GraphConfig):
        self.config = config
        
        # Node2Vec parameters
        self.embed_dim = config.node2vec_dim
        self.walk_length = config.node2vec_walk_length
        self.num_walks = config.node2vec_num_walks
        self.p = config.node2vec_p  # Return parameter
        self.q = config.node2vec_q  # In-out parameter
        
        # Trained embeddings
        self.embeddings = None
        self._is_fitted = False
        
    def fit_transform(self, graph: Data) -> torch.Tensor:
        """
        Generate Node2Vec embeddings from graph structure.
        
        Parameters
        ----------
        graph : Data
            PyTorch Geometric graph data object.
            
        Returns
        -------
        torch.Tensor
            Node embeddings of shape (n_nodes, embed_dim).
        """
        # Convert to NetworkX for random walk generation
        nx_graph = self._convert_to_networkx(graph)
        
        # Generate random walks
        walks = self._generate_walks(nx_graph)
        
        # Train Word2Vec model on walks
        embeddings = self._train_embeddings(walks)
        
        self.embeddings = embeddings
        self._is_fitted = True
        
        return embeddings
    
    def _convert_to_networkx(self, graph: Data) -> nx.Graph:
        """
        Convert PyTorch Geometric graph to NetworkX.
        
        Parameters
        ----------
        graph : Data
            PyTorch Geometric graph.
            
        Returns
        -------
        nx.Graph
            NetworkX graph object.
        """
        # Convert to NetworkX
        nx_graph = to_networkx(graph, to_undirected=True)
        
        # Add edge weights if available
        if graph.edge_attr is not None:
            edge_weights = graph.edge_attr.cpu().numpy()
            edge_list = graph.edge_index.t().cpu().numpy()
            
            for i, (u, v) in enumerate(edge_list):
                if nx_graph.has_edge(u, v):
                    nx_graph[u][v]['weight'] = float(edge_weights[i])
                    
        # Set default weights for edges without weights
        for u, v in nx_graph.edges():
            if 'weight' not in nx_graph[u][v]:
                nx_graph[u][v]['weight'] = 1.0
                
        return nx_graph
    
    def _generate_walks(self, graph: nx.Graph) -> List[List[str]]:
        """
        Generate biased random walks using Node2Vec strategy.
        
        Parameters
        ----------
        graph : nx.Graph
            NetworkX graph for walk generation.
            
        Returns
        -------
        List[List[str]]
            List of random walks, where each walk is a list of node IDs as strings.
        """
        # Precompute transition probabilities
        alias_nodes, alias_edges = self._preprocess_transition_probs(graph)
        
        walks = []
        nodes = list(graph.nodes())
        
        # Generate multiple walks from each node
        for walk_iter in range(self.num_walks):
            np.random.shuffle(nodes)  # Randomize starting order
            
            for node in nodes:
                walk = self._node2vec_walk(
                    graph, node, alias_nodes, alias_edges
                )
                walks.append(walk)
                
        return walks
    
    def _node2vec_walk(self, graph: nx.Graph, start_node: int,
                      alias_nodes: dict, alias_edges: dict) -> List[str]:
        """
        Generate a single Node2Vec random walk.
        
        Parameters
        ----------
        graph : nx.Graph
            NetworkX graph.
        start_node : int
            Starting node for the walk.
        alias_nodes : dict
            Precomputed alias tables for nodes.
        alias_edges : dict
            Precomputed alias tables for edges.
            
        Returns
        -------
        List[str]
            Random walk as list of node IDs (as strings).
        """
        walk = [str(start_node)]
        
        while len(walk) < self.walk_length:
            cur = int(walk[-1])
            cur_nbrs = list(graph.neighbors(cur))
            
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    # First step: uniform random selection
                    next_node = cur_nbrs[self._alias_draw(alias_nodes[cur])]
                else:
                    # Subsequent steps: biased selection based on p, q
                    prev = int(walk[-2])
                    edge_key = (prev, cur)
                    
                    if edge_key in alias_edges:
                        next_node = cur_nbrs[self._alias_draw(alias_edges[edge_key])]
                    else:
                        # Fallback to uniform selection
                        next_node = np.random.choice(cur_nbrs)
                        
                walk.append(str(next_node))
            else:
                # Dead end: terminate walk
                break
                
        return walk
    
    def _preprocess_transition_probs(self, graph: nx.Graph) -> Tuple[dict, dict]:
        """
        Preprocess transition probabilities for efficient sampling.
        
        Parameters
        ----------
        graph : nx.Graph
            NetworkX graph.
            
        Returns
        -------
        Tuple[dict, dict]
            Alias tables for nodes and edges.
        """
        alias_nodes = {}
        alias_edges = {}
        
        # Precompute alias tables for each node
        for node in graph.nodes():
            neighbors = list(graph.neighbors(node))
            if len(neighbors) > 0:
                # Get edge weights
                weights = [graph[node][nbr].get('weight', 1.0) for nbr in neighbors]
                alias_nodes[node] = self._alias_setup(weights)
            else:
                alias_nodes[node] = ([], [])
                
        # Precompute alias tables for each edge (for biased walks)
        for edge in graph.edges():
            u, v = edge
            alias_edges[(u, v)] = self._get_alias_edge(graph, u, v)
            alias_edges[(v, u)] = self._get_alias_edge(graph, v, u)
            
        return alias_nodes, alias_edges
    
    def _get_alias_edge(self, graph: nx.Graph, src: int, dst: int) -> Tuple[List[int], List[float]]:
        """
        Get alias table for a specific edge transition.
        
        Parameters
        ----------
        graph : nx.Graph
            NetworkX graph.
        src : int
            Source node.
        dst : int
            Destination node.
            
        Returns
        -------
        Tuple[List[int], List[float]]
            Alias table for the edge.
        """
        unnormalized_probs = []
        
        for dst_nbr in graph.neighbors(dst):
            weight = graph[dst][dst_nbr].get('weight', 1.0)
            
            if dst_nbr == src:
                # Return to previous node
                unnormalized_probs.append(weight / self.p)
            elif graph.has_edge(dst_nbr, src):
                # Stay in local neighborhood
                unnormalized_probs.append(weight)
            else:
                # Move to distant node
                unnormalized_probs.append(weight / self.q)
                
        return self._alias_setup(unnormalized_probs)
    
    def _alias_setup(self, probs: List[float]) -> Tuple[List[int], List[float]]:
        """
        Create alias table for efficient non-uniform sampling.
        
        Based on the alias method for discrete sampling.
        
        Parameters
        ----------
        probs : List[float]
            Unnormalized probabilities.
            
        Returns
        -------
        Tuple[List[int], List[float]]
            Alias table (J, q arrays).
        """
        if len(probs) == 0:
            return [], []
            
        # Normalize probabilities
        probs = np.array(probs, dtype=np.float64)
        probs = probs / np.sum(probs)
        
        K = len(probs)
        q = np.zeros(K, dtype=np.float64)
        J = np.zeros(K, dtype=np.int32)
        
        # Scale probabilities
        for i, prob in enumerate(probs):
            q[i] = K * prob
            
        # Create alias table
        smaller = []
        larger = []
        
        for i, qi in enumerate(q):
            if qi < 1.0:
                smaller.append(i)
            else:
                larger.append(i)
                
        while smaller and larger:
            small = smaller.pop()
            large = larger.pop()
            
            J[small] = large
            q[large] = q[large] + q[small] - 1.0
            
            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)
                
        return J.tolist(), q.tolist()
    
    def _alias_draw(self, alias_table: Tuple[List[int], List[float]]) -> int:
        """
        Draw sample from alias table.
        
        Parameters
        ----------
        alias_table : Tuple[List[int], List[float]]
            Alias table (J, q arrays).
            
        Returns
        -------
        int
            Sampled index.
        """
        J, q = alias_table
        
        if len(J) == 0:
            return 0
            
        K = len(J)
        kk = int(np.floor(np.random.rand() * K))
        
        if np.random.rand() < q[kk]:
            return kk
        else:
            return J[kk]
    
    def _train_embeddings(self, walks: List[List[str]]) -> torch.Tensor:
        """
        Train Word2Vec model on random walks to generate embeddings.
        
        Parameters
        ----------
        walks : List[List[str]]
            List of random walks.
            
        Returns
        -------
        torch.Tensor
            Node embeddings.
        """
        if len(walks) == 0:
            warnings.warn("No walks generated. Returning zero embeddings.")
            return torch.zeros((1, self.embed_dim))
            
        # Determine vocabulary size (number of unique nodes)
        all_nodes = set()
        for walk in walks:
            all_nodes.update(walk)
            
        vocab_size = len(all_nodes)
        
        if vocab_size == 0:
            warnings.warn("Empty vocabulary. Returning zero embeddings.")
            return torch.zeros((1, self.embed_dim))
            
        # Train Word2Vec model
        try:
            model = Word2Vec(
                walks,
                vector_size=self.embed_dim,
                window=10,  # Context window size
                min_count=1,  # Minimum word frequency
                workers=4,  # Number of worker threads
                sg=1,  # Skip-gram model
                epochs=5
            )
            
            # Extract embeddings for all nodes
            embeddings = np.zeros((vocab_size, self.embed_dim))
            node_to_idx = {}
            
            for i, node in enumerate(sorted(all_nodes, key=int)):
                node_to_idx[node] = i
                if node in model.wv:
                    embeddings[i] = model.wv[node]
                else:
                    # Random initialization for nodes not in vocabulary
                    embeddings[i] = np.random.normal(0, 0.1, self.embed_dim)
                    
        except Exception as e:
            warnings.warn(f"Word2Vec training failed: {e}. Using random embeddings.")
            embeddings = np.random.normal(0, 0.1, (vocab_size, self.embed_dim))
            
        return torch.from_numpy(embeddings.astype(np.float32))
    
    def transform(self, graph: Data) -> torch.Tensor:
        """
        Transform new graph using fitted embeddings.
        
        Parameters
        ----------
        graph : Data
            New graph to transform.
            
        Returns
        -------
        torch.Tensor
            Node embeddings for the new graph.
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before transform. Use fit_transform instead.")
            
        # For simplicity, return fitted embeddings
        # In practice, you might want to handle new nodes differently
        return self.embeddings