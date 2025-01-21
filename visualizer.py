from __future__ import annotations

import logging
import json
import sys
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple, NamedTuple, Optional, Union, Set, Any
from enum import Enum, auto
import numpy as np
from scipy import sparse, ndimage, stats, linalg
from collections import defaultdict
import hashlib
from scipy.io import mmread
import pandas as pd
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
import argparse


class VisualizationType(Enum):
  """Types of visualizations available"""
  BASIC = auto()
  DETAILED = auto()
  PATTERN = auto()
  STATISTICAL = auto()
  ALL = auto()

@dataclass
class AnalyzerConfig:
  """Configuration for matrix analysis"""
  input_dir: Path
  output_dir: Optional[Path] = None
  metadata_file: str = "matrix_metadata.csv"
  
  # base analysis settings
  max_matrix_size: int = 50000
  chunk_size: int = 1000
  parallel_workers: int = 1 # 4
  log_level: str = "INFO"
  analyze_patterns: bool = False
  
  # visualization settings
  visualization_types: List[VisualizationType] = field( default_factory=lambda: [VisualizationType.BASIC] )
  figure_size: tuple[float, float] = (15, 10)
  dpi: int = 300
  style: str = "darkgrid"
  color_palette: str = "deep"
  
  # pattern analysis settings
  min_block_size: int = 2
  max_block_size: int = 8
  min_pattern_frequency: float = 0.01
  ignore_zero_patterns: bool = False
  max_patterns_display: int = 10
  
  # statistical analysis settings
  significance_level: float = 0.05
  sample_size: Optional[int] = None
  enable_advanced_stats: bool = False
  
  @classmethod
  def from_dict(cls, config_dict: Dict[str, Any]) -> AnalyzerConfig:
    """Create config from dictionary"""
    if 'input_dir' in config_dict:
      config_dict['input_dir'] = Path(config_dict['input_dir'])
    if 'output_dir' in config_dict and config_dict['output_dir']:
      config_dict['output_dir'] = Path(config_dict['output_dir'])
        
    if 'visualization_types' in config_dict:
      config_dict['visualization_types'] = [
        VisualizationType[v] if isinstance(v, str) else v 
        for v in config_dict['visualization_types']
      ]
        
    return cls(**config_dict)
  
  @classmethod
  def from_json(cls, json_path: Union[str, Path]) -> AnalyzerConfig:
    """Load config from JSON file"""
    with open(json_path, 'r') as f:
      config_dict = json.load(f)
    return cls.from_dict(config_dict)
  
  def to_dict(self) -> Dict[str, Any]:
    """Convert config to dictionary"""
    config_dict = asdict(self)
    config_dict['input_dir'] = str(config_dict['input_dir'])
    if config_dict['output_dir']:
      config_dict['output_dir'] = str(config_dict['output_dir'])
    config_dict['visualization_types'] = [v.name for v in config_dict['visualization_types']]
    return config_dict
  
  def to_json(self, json_path: Union[str, Path]) -> None:
    """Save config to JSON file"""
    with open(json_path, 'w') as f:
      json.dump(self.to_dict(), f, indent=4)

  def validate(self) -> None:
    """Validate configuration settings"""
    if not self.input_dir.exists():
      raise ValueError(f"Input directory does not exist: {self.input_dir}")
        
    if self.output_dir is None:
      self.output_dir = self.input_dir / 'visualizations'
        
    if self.max_block_size < self.min_block_size:
      raise ValueError("max_block_size must be greater than min_block_size")
        
    if not 0 <= self.min_pattern_frequency <= 1:
      raise ValueError("min_pattern_frequency must be between 0 and 1")
        
    if self.parallel_workers < 1:
      raise ValueError("parallel_workers must be positive")

class Logger:
  """Centralized logging configuration"""
  
  @staticmethod
  def setup(name: str, level: str = "INFO") -> logging.Logger:
    """Set up a logger with consistent formatting"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
      logger.setLevel(getattr(logging, level))
      
      console_handler = logging.StreamHandler()
      console_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
      )
      logger.addHandler(console_handler)
      
      file_handler = logging.FileHandler(f"{name}.log")
      file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
      )
      logger.addHandler(file_handler)
    
    return logger

class MatrixAnalyzerException(Exception):
  """Base exception for matrix analyzer errors"""
  pass

class ConfigurationError(MatrixAnalyzerException):
  """Configuration related errors"""
  pass

class AnalysisError(MatrixAnalyzerException):
  """Analysis related errors"""
  pass

class VisualizationError(MatrixAnalyzerException):
  """Visualization related errors"""
  pass



### PATTERN ANALYSIS MODULE ###

@dataclass
class PatternMetrics:
  """Detailed metrics for a matrix pattern"""
  pattern: np.ndarray
  frequency: float
  density: float
  symmetry_score: float
  connectivity: float
  spatial_correlation: float
  hierarchical_level: int
  variants: List[np.ndarray] = field(default_factory=list)
  optimization_potential: float = 0.0
  
  def to_dict(self) -> Dict:
    """Convert metrics to dictionary format"""
    return {
      'density': self.density,
      'frequency': self.frequency,
      'symmetry_score': self.symmetry_score,
      'connectivity': self.connectivity,
      'spatial_correlation': self.spatial_correlation,
      'hierarchical_level': self.hierarchical_level,
      'optimization_potential': self.optimization_potential
    }

class PatternAnalyzer:
  """Advanced pattern analysis with optimization insights"""
  
  def __init__(self,
    min_block_size: int = 2,
    max_block_size: int = 8,
    min_frequency: float = 0.01,
    ignore_zeros: bool = False,
    similarity_threshold: float = 0.85,
    enable_hierarchical: bool = True,
    n_workers: int = 4,
  ):
    """Initialize pattern analyzer with configuration"""
    self.min_block_size = min_block_size
    self.max_block_size = max_block_size
    self.min_frequency = min_frequency
    self.ignore_zeros = ignore_zeros
    self.similarity_threshold = similarity_threshold
    self.enable_hierarchical = enable_hierarchical
    self.n_workers = n_workers
    self.pattern_cache = {}
    self.logger = logging.getLogger(__name__)
  
  def _compute_pattern_hash(self, pattern: np.ndarray) -> str:
    """Compute stable hash for pattern identification"""
    canonical_pattern = self._canonicalize_pattern(pattern)
    return hashlib.sha256(canonical_pattern.tobytes()).hexdigest()
  
  def _canonicalize_pattern(self, pattern: np.ndarray) -> np.ndarray:
    """Convert pattern to canonical form for comparison"""
    binary = (pattern != 0).astype(np.uint8)
    variants = [
      binary,
      np.fliplr(binary),
      np.flipud(binary),
      np.rot90(binary),
      np.rot90(binary, 2),
      np.rot90(binary, 3)
    ]
    return min(variants, key=lambda x: x.tobytes()) # lexicographically minimal variant
  
  def _generate_pattern_variants(self, pattern: np.ndarray) -> List[np.ndarray]:
    """Generate all valid pattern variants"""
    variants = []
    pattern = pattern.copy()
    
    for k in range(4):
      rotated = np.rot90(pattern, k)
      variants.append(rotated)
      variants.append(np.fliplr(rotated))
      variants.append(np.flipud(rotated))
    
    return list({arr.tobytes(): arr for arr in variants}.values())
  
  def _compute_pattern_metrics(self, 
    pattern: np.ndarray,
    frequency: float,
    level: int,
  ) -> PatternMetrics:
    """Compute comprehensive pattern metrics"""
    binary_pattern = (pattern != 0).astype(np.uint8)
    
    density = np.count_nonzero(pattern) / pattern.size
    
    symmetry_score = self._compute_symmetry(pattern)
    
    connectivity = self._compute_connectivity(binary_pattern)
    
    spatial_correlation = self._compute_spatial_correlation(pattern)
    
    variants = self._generate_pattern_variants(pattern)
    
    optimization_potential = self._compute_optimization_potential(
      density, symmetry_score, connectivity, spatial_correlation
    )
    
    return PatternMetrics(
      pattern=pattern,
      frequency=frequency,
      density=density,
      symmetry_score=symmetry_score,
      connectivity=connectivity,
      spatial_correlation=spatial_correlation,
      hierarchical_level=level,
      variants=variants,
      optimization_potential=optimization_potential
    )
  
  def _compute_symmetry(self, pattern: np.ndarray) -> float:
    """Compute pattern symmetry score"""
    scores = []
    
    scores.append(np.mean(pattern == np.fliplr(pattern)))
    
    scores.append(np.mean(pattern == np.flipud(pattern)))
    
    if pattern.shape[0] == pattern.shape[1]:
      scores.append(np.mean(pattern == pattern.T))
      scores.append(np.mean(pattern == np.rot90(pattern)))
    
    return np.mean(scores)
  
  def _compute_connectivity(self, binary_pattern: np.ndarray) -> float:
    """Compute pattern connectivity score"""
    labeled, num_features = ndimage.label(binary_pattern)
    if num_features == 0:
      return 0.0
    
    sizes = np.bincount(labeled.ravel())[1:]
    
    return float(np.max(sizes)) / np.sum(sizes)
  
  def _compute_spatial_correlation(self, pattern: np.ndarray) -> float:
    """Compute spatial correlation score"""
    if pattern.size <= 1:
      return 0.0
        
    pattern_norm = pattern - np.mean(pattern)
    variance = np.var(pattern)
    
    if variance == 0:
      return 0.0
        
    correlation = ndimage.correlate(pattern_norm, pattern_norm, mode='constant')
    max_corr = np.max(np.abs(correlation))
    
    return max_corr / (variance * pattern.size)
  
  def _compute_optimization_potential(self,
    density: float,
    symmetry: float,
    connectivity: float,
    spatial_correlation: float,
  ) -> float:
    """Compute potential for optimization based on pattern properties"""
    weights = { # weight factors for different properties (???)
      'density': 0.3,
      'symmetry': 0.2,
      'connectivity': 0.3,
      'spatial_correlation': 0.2
    }
    
    score = (
      weights['density'] * (1 - abs(0.5 - density) * 2) +  # target ~50% density (???)
      weights['symmetry'] * symmetry +
      weights['connectivity'] * connectivity +
      weights['spatial_correlation'] * spatial_correlation
    )
    
    return float(score)
  
  def analyze_blocks(self, matrix: sparse.spmatrix, block_size: int) -> Dict[str, Tuple[PatternMetrics, List[Tuple[int, int]]]]:
    """Analyze matrix blocks of specific size"""
    rows, cols = matrix.shape
    n_blocks_row = rows // block_size
    n_blocks_col = cols // block_size
    
    patterns = defaultdict(list)
    metrics_cache = {}
    
    for i in range(n_blocks_row):
      row_start = i * block_size
      row_end = row_start + block_size
      
      for j in range(n_blocks_col):
        col_start = j * block_size
        col_end = col_start + block_size
        
        block = matrix[row_start:row_end, col_start:col_end].toarray()
        if not np.any(block) and self.ignore_zeros:
          continue
        
        pattern_hash = self._compute_pattern_hash(block)
        patterns[pattern_hash].append((row_start, col_start))
    
    total_blocks = n_blocks_row * n_blocks_col
    result = {}
    
    for pattern_hash, locations in patterns.items():
      frequency = len(locations) / total_blocks
      if frequency >= self.min_frequency:
        if pattern_hash not in metrics_cache:
          block = matrix[locations[0][0]:locations[0][0]+block_size, locations[0][1]:locations[0][1]+block_size].toarray()
          metrics = self._compute_pattern_metrics(block, frequency, block_size)
          metrics_cache[pattern_hash] = metrics
        
        result[pattern_hash] = (metrics_cache[pattern_hash], locations)
    
    return result
  
  def analyze_matrix(self, matrix: sparse.spmatrix) -> Dict[int, Dict[str, Tuple[PatternMetrics, List[Tuple[int, int]]]]]:
    """Perform complete pattern analysis of matrix"""
    results = {}
    
    with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
      future_to_size = {
        executor.submit(self.analyze_blocks, matrix, size): size
        for size in range(self.min_block_size, self.max_block_size + 1)
      }
      
      for future in future_to_size:
        size = future_to_size[future]
        try:
          patterns = future.result()
          if patterns:
            results[size] = patterns
        except Exception as e:
          self.logger.error(f"Error analyzing block size {size}: {str(e)}")
    
    return results
  
  def find_optimization_patterns(self, matrix: sparse.spmatrix, min_optimization_score: float = 0.6) -> List[Tuple[PatternMetrics, List[Tuple[int, int]]]]:
    """Find patterns suitable for optimization"""
    all_patterns = self.analyze_matrix(matrix)
    optimization_candidates = []
    
    for size, patterns in all_patterns.items():
      for pattern_hash, (metrics, locations) in patterns.items():
        if metrics.optimization_potential >= min_optimization_score:
          optimization_candidates.append((metrics, locations))
    
    return sorted(optimization_candidates, key=lambda x: x[0].optimization_potential, reverse=True)
  
  def suggest_optimization_strategy(self, pattern_metrics: PatternMetrics) -> str:
    """Suggest optimization strategy based on pattern properties"""
    suggestions = []
    
    if pattern_metrics.symmetry_score > 0.8:
      suggestions.append("Use symmetric multiplication optimizations")
    
    if pattern_metrics.density < 0.3:
      suggestions.append("Apply sparse-sparse multiplication techniques")
    elif pattern_metrics.density > 0.7:
      suggestions.append("Use dense multiplication algorithms")
    else:
      suggestions.append("Consider hybrid sparse-dense methods")
    
    if pattern_metrics.connectivity > 0.8:
      suggestions.append("Exploit connected block structure")
    
    if pattern_metrics.spatial_correlation > 0.7:
      suggestions.append("Use spatial locality optimizations")
    
    if pattern_metrics.hierarchical_level > 2:
      suggestions.append("Apply hierarchical multiplication strategy")
    
    return " | ".join(suggestions)


### STATISTICAL ANALYSIS MODULE ###


@dataclass
class StatisticalMetrics:
    """Comprehensive statistical metrics for matrix analysis"""
    
    skewness: float
    kurtosis: float
    entropy: float
    normality_stats: Dict[str, float]
    
    spatial_correlation: float
    value_correlation: float
    row_correlation: float
    col_correlation: float
    
    sparsity_pattern_entropy: float
    block_structure_score: float
    locality_score: float
    
    memory_access_pattern: Dict[str, float]
    computational_density: float
    
    def to_dict(self) -> Dict:
      """Convert metrics to dictionary format"""
      return {
        'distribution_metrics': {
          'skewness': self.skewness,
          'kurtosis': self.kurtosis,
          'entropy': self.entropy,
          'normality_stats': self.normality_stats
        },
        'correlation_metrics': {
          'spatial_correlation': self.spatial_correlation,
          'value_correlation': self.value_correlation,
          'row_correlation': self.row_correlation,
          'col_correlation': self.col_correlation
        },
        'structural_metrics': {
          'sparsity_pattern_entropy': self.sparsity_pattern_entropy,
          'block_structure_score': self.block_structure_score,
          'locality_score': self.locality_score
        },
        'performance_metrics': {
          'memory_access_pattern': self.memory_access_pattern,
          'computational_density': self.computational_density
        }
      }

class StatisticalAnalyzer:
    """Advanced statistical analysis for sparse matrices"""
    
    def __init__(self, 
      significance_level: float = 0.05,
      block_sizes: List[int] = None,
      enable_advanced_analysis: bool = True,
    ):  
      """Initialize the statistical analyzer"""
      self.significance_level = significance_level
      self.block_sizes = block_sizes or [2, 4, 8, 16, 32]
      self.enable_advanced_analysis = enable_advanced_analysis
      self.logger = logging.getLogger(__name__)
    
    def analyze_matrix(self, matrix: sparse.spmatrix) -> StatisticalMetrics:
      """Perform comprehensive statistical analysis of a matrix"""
      try:
        distribution_metrics = self._analyze_distribution(matrix)
        
        correlation_metrics = self._analyze_correlations(matrix)
        
        structural_metrics = self._analyze_structure(matrix)
        
        performance_metrics = self._analyze_performance_characteristics(matrix)
        
        return StatisticalMetrics(
          # distribution metrics
          skewness=distribution_metrics['skewness'],
          kurtosis=distribution_metrics['kurtosis'],
          entropy=distribution_metrics['entropy'],
          normality_stats=distribution_metrics['normality_stats'],
          
          # correlation metrics
          spatial_correlation=correlation_metrics['spatial'],
          value_correlation=correlation_metrics['value'],
          row_correlation=correlation_metrics['row'],
          col_correlation=correlation_metrics['col'],
          
          # structural metrics
          sparsity_pattern_entropy=structural_metrics['pattern_entropy'],
          block_structure_score=structural_metrics['block_score'],
          locality_score=structural_metrics['locality'],
          
          # performance metrics
          memory_access_pattern=performance_metrics['access_pattern'],
          computational_density=performance_metrics['comp_density']
        )
          
      except Exception as e:
        self.logger.error(f"Error during statistical analysis: {str(e)}")
        raise
    
    def _analyze_distribution(self, matrix: sparse.spmatrix) -> Dict[str, Any]:
      """Analyze value distribution characteristics"""
      data = matrix.data
      
      with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        skewness = float(stats.skew(data))
        kurtosis = float(stats.kurtosis(data))
      
      hist, bin_edges = np.histogram(data, bins='auto', density=True)
      entropy = float(stats.entropy(hist)) if len(hist) > 0 else 0.0
      
      if len(data) >= 8:
        statistic, p_value = stats.normaltest(data)
      else:
        statistic, p_value = 0.0, 1.0
          
      normality_stats = {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'is_normal': p_value > self.significance_level
      }
      
      return {
        'skewness': skewness,
        'kurtosis': kurtosis,
        'entropy': entropy,
        'normality_stats': normality_stats
      }
    
    def _analyze_correlations(self, matrix: sparse.spmatrix) -> Dict[str, float]:
      """Analyze various types of correlations in the matrix"""
      rows, cols = matrix.nonzero()
      if len(rows) > 0:
        spatial_correlation = self._compute_spatial_correlation(rows, cols, matrix.shape)
      else:
        spatial_correlation = 0.0
      
      value_correlation = self._compute_value_correlation(matrix)
      
      row_correlation = self._compute_row_correlation(matrix)
      col_correlation = self._compute_col_correlation(matrix)
      
      return {
        'spatial': spatial_correlation,
        'value': value_correlation,
        'row': row_correlation,
        'col': col_correlation
      }
    
    def _compute_spatial_correlation(self, rows: np.ndarray, cols: np.ndarray, shape: Tuple[int, int]) -> float:
      """Compute spatial correlation using Moran's I statistic"""
      if len(rows) < 2:
        return 0.0
      
      coords = np.column_stack((rows, cols))
      dists = np.zeros((len(rows), len(rows)))
      
      for i in range(len(rows)):
        dists[i] = np.sqrt(np.sum((coords - coords[i])**2, axis=1))
      
      max_dist = np.sqrt(shape[0]**2 + shape[1]**2)
      dists /= max_dist
      
      mean_dist = np.mean(dists[np.triu_indices_from(dists, k=1)])
      variance = np.var(dists[np.triu_indices_from(dists, k=1)])
      
      if variance == 0:
        return 0.0
          
      n = len(rows)
      moran_i = (n / np.sum(dists != 0) * np.sum(dists * (dists - mean_dist)) /  (variance * (n - 1)))
      
      return float(moran_i)
    
    def _compute_value_correlation(self, matrix: sparse.spmatrix) -> float:
      """Compute correlation between neighboring non-zero values"""
      if matrix.nnz < 2:
          return 0.0
          
      data = matrix.data
      correlations = []
      
      for i in range(len(data)-1):
        correlations.append(data[i] * data[i+1])
      
      return float(np.mean(correlations) / (np.std(data) ** 2)) if correlations else 0.0
    
    def _compute_row_correlation(self, matrix: sparse.spmatrix) -> float:
      """Compute correlation between row patterns"""
      row_patterns = np.diff(matrix.indptr)
      if len(row_patterns) < 2:
        return 0.0
          
      return float(np.corrcoef(row_patterns[:-1], row_patterns[1:])[0, 1])
    
    def _compute_col_correlation(self, matrix: sparse.spmatrix) -> float:
      """Compute correlation between column patterns"""
      col_patterns = np.bincount(matrix.indices, minlength=matrix.shape[1])
      if len(col_patterns) < 2:
        return 0.0
          
      return float(np.corrcoef(col_patterns[:-1], col_patterns[1:])[0, 1])
    
    def _analyze_structure(self, matrix: sparse.spmatrix) -> Dict[str, float]:
      """Analyze structural characteristics of the matrix"""
      pattern_entropy = self._compute_pattern_entropy(matrix)
      
      block_score = self._analyze_block_structure(matrix)
      
      locality = self._compute_locality_score(matrix)
      
      return {
        'pattern_entropy': pattern_entropy,
        'block_score': block_score,
        'locality': locality
      }
  
    def _compute_pattern_entropy(self, matrix: sparse.spmatrix) -> float:
      """Compute entropy of the sparsity pattern"""
      rows, cols = matrix.shape
      pattern = (matrix != 0).astype(int)
      
      window_size = min(32, min(rows, cols))
      patterns = []
      
      for i in range(0, rows - window_size + 1, window_size // 2):
        for j in range(0, cols - window_size + 1, window_size // 2):
          window = pattern[i:i+window_size, j:j+window_size]
          patterns.append(str(window.tobytes()))
      
      pattern_counts = defaultdict(int)
      for p in patterns:
        pattern_counts[p] += 1
          
      probabilities = np.array(list(pattern_counts.values())) / len(patterns)
      return float(stats.entropy(probabilities))
    
    def _analyze_block_structure(self, matrix: sparse.spmatrix) -> float:
      """Analyze block structure characteristics"""
      block_densities = []
      
      for size in self.block_sizes:
        if min(matrix.shape) < size:
          continue
            
        rows, cols = matrix.shape
        n_blocks_row = rows // size
        n_blocks_col = cols // size
        
        for i in range(n_blocks_row):
          for j in range(n_blocks_col):
            block = matrix[i*size:(i+1)*size, j*size:(j+1)*size]
            block_densities.append(block.nnz / (size * size))
      
      if not block_densities:
        return 0.0
          
      return float(1 - np.std(block_densities))
    
    def _compute_locality_score(self, matrix: sparse.spmatrix) -> float:
      """Compute locality score based on non-zero element clustering"""
      if matrix.nnz == 0:
        return 0.0
          
      rows, cols = matrix.nonzero()
      
      distances = np.sqrt(np.diff(rows)**2 + np.diff(cols)**2)
      
      if len(distances) == 0:
        return 0.0
          
      max_distance = np.sqrt(matrix.shape[0]**2 + matrix.shape[1]**2)
      normalized_distances = distances / max_distance
      
      return float(1 / (1 + np.mean(normalized_distances)))
    
    def _analyze_performance_characteristics(self, matrix: sparse.spmatrix) -> Dict[str, Any]:
      """Analyze characteristics relevant to computational performance"""
      access_pattern = self._analyze_memory_access(matrix)
      
      comp_density = self._compute_computational_density(matrix)
      
      return {
        'access_pattern': access_pattern,
        'comp_density': comp_density
      }
    
    def _analyze_memory_access(self, matrix: sparse.spmatrix) -> Dict[str, float]:
      """Analyze memory access patterns"""
      row_jumps = np.diff(matrix.indices)
      row_locality = 1 / (1 + np.mean(np.abs(row_jumps))) if len(row_jumps) > 0 else 0
      
      col_accesses = np.bincount(matrix.indices, minlength=matrix.shape[1])
      col_balance = 1 - np.std(col_accesses) / np.mean(col_accesses) if np.mean(col_accesses) > 0 else 0
      
      return {
        'row_locality': float(row_locality),
        'col_balance': float(col_balance)
      }
    
    def _compute_computational_density(self, matrix: sparse.spmatrix) -> float:
      """Compute computational density metric"""
      if matrix.nnz == 0:
        return 0.0
          
      basic_density = matrix.nnz / (matrix.shape[0] * matrix.shape[1])
      
      row_work = np.diff(matrix.indptr)
      work_balance = 1 - np.std(row_work) / np.mean(row_work) if np.mean(row_work) > 0 else 0
      
      return float(np.sqrt(basic_density * work_balance))


### VISUALIZATION MODULE ###



@dataclass
class VisualizationConfig:
  """Configuration for visualization settings"""
  style: str = "darkgrid"
  palette: str = "deep"
  figure_size: Tuple[float, float] = (15, 10)
  dpi: int = 300
  font_family: str = "sans-serif"
  font_size: int = 10
  show_grid: bool = True
  color_map: str = "viridis"
  save_format: str = "png"
    
  def apply(self):
    """Apply visualization configuration"""
    sns.set_theme(style=self.style, palette=self.palette)
    plt.rcParams.update({
      'figure.figsize': self.figure_size,
      'figure.dpi': self.dpi,
      'font.family': self.font_family,
      'font.size': self.font_size,
      'axes.grid': self.show_grid
    })

class MatrixVisualizer:
  """Comprehensive matrix visualization system"""
  
  def __init__(self, config: Optional[VisualizationConfig] = None):
    """Initialize visualizer with configuration"""
    self.config = config or VisualizationConfig()
    self.config.apply()
    self.logger = logging.getLogger(__name__)
      
  def create_visualization(self, matrix: sparse.spmatrix,stats: Any, filename: str, output_dir: Path, viz_type: str = "basic") -> None:
    """Create and save matrix visualization"""
    try:
      if viz_type == "basic":
        fig = self._create_basic_visualization(matrix, stats)
      elif viz_type == "detailed":
        fig = self._create_detailed_visualization(matrix, stats)
      elif viz_type == "pattern":
        fig = self._create_pattern_visualization(matrix, stats)
      elif viz_type == "statistical":
        fig = self._create_statistical_visualization(matrix, stats)
      else:
        raise ValueError(f"Unknown visualization type: {viz_type}")
      
      fig.suptitle(f"Matrix Analysis: {filename}", y=1.02, fontsize=14)
      plt.tight_layout()
      
      output_path = output_dir / f"{Path(filename).stem}_{viz_type}.{self.config.save_format}"
      fig.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
      plt.close(fig)
      
      self.logger.info(f"Saved {viz_type} visualization to {output_path}")
      
    except Exception as e:
      self.logger.error(f"Error creating visualization: {str(e)}")
      raise
  
  def _create_basic_visualization(self,matrix: sparse.spmatrix,stats: Any) -> Figure:
    """Create basic visualization with essential information"""
    fig = plt.figure(figsize=self.config.figure_size)
    gs = GridSpec(2, 2, figure=fig)
    
    ax_structure = fig.add_subplot(gs[0, 0])
    self._plot_matrix_structure(matrix, ax_structure)
    
    ax_values = fig.add_subplot(gs[0, 1])
    self._plot_value_distribution(matrix, ax_values)
    
    ax_stats = fig.add_subplot(gs[1, :])
    self._plot_basic_stats(stats, ax_stats)
    
    return fig
  
  def _create_detailed_visualization(self,matrix: sparse.spmatrix,stats: Any) -> Figure:
    """Create detailed visualization with comprehensive analysis"""
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 0.8])
    
    ax_structure = fig.add_subplot(gs[0, 0:2])
    self._plot_matrix_structure(matrix, ax_structure)
    
    ax_values = fig.add_subplot(gs[0, 2])
    self._plot_value_distribution(matrix, ax_values)
    
    ax_nonzeros = fig.add_subplot(gs[1, 0])
    self._plot_nonzero_distribution(matrix, ax_nonzeros)
    
    ax_blocks = fig.add_subplot(gs[1, 1])
    self._plot_block_density(matrix, ax_blocks)
    
    ax_freqs = fig.add_subplot(gs[1, 2])
    self._plot_value_frequencies(matrix, ax_freqs)
    
    ax_stats = fig.add_subplot(gs[2, :])
    self._plot_detailed_stats(stats, ax_stats)
    
    return fig
  
  def _create_pattern_visualization(self,matrix: sparse.spmatrix,stats: Any) -> Figure:
    """Create visualization focusing on pattern analysis"""
    fig = plt.figure(figsize=self.config.figure_size)
    gs = GridSpec(2, 2, figure=fig)
    
    ax_pattern_dist = fig.add_subplot(gs[0, 0])
    self._plot_pattern_distribution(stats, ax_pattern_dist)
    
    ax_patterns = fig.add_subplot(gs[0, 1])
    self._plot_pattern_examples(stats, ax_patterns)
    
    ax_metrics = fig.add_subplot(gs[1, 0])
    self._plot_pattern_metrics(stats, ax_metrics)
    
    ax_suggestions = fig.add_subplot(gs[1, 1])
    self._plot_optimization_suggestions(stats, ax_suggestions)
    
    return fig
  
  def _create_statistical_visualization(self,matrix: sparse.spmatrix,stats: Any) -> Figure:
    """Create visualization focusing on statistical analysis"""
    fig = plt.figure(figsize=self.config.figure_size)
    gs = GridSpec(2, 2, figure=fig)
    
    ax_dist = fig.add_subplot(gs[0, 0])
    self._plot_distribution_metrics(stats, ax_dist)
    
    ax_corr = fig.add_subplot(gs[0, 1])
    self._plot_correlation_metrics(stats, ax_corr)
    
    ax_struct = fig.add_subplot(gs[1, 0])
    self._plot_structural_metrics(stats, ax_struct)
    
    ax_perf = fig.add_subplot(gs[1, 1])
    self._plot_performance_metrics(stats, ax_perf)
    
    return fig
  
  def _plot_matrix_structure(self, matrix: sparse.spmatrix,ax: Axes) -> None:
    """Plot matrix sparsity pattern"""
    ax.spy(matrix, markersize=0.5, color='#2E86C1', alpha=0.6)
    ax.set_title("Matrix Structure", pad=10)
    ax.set_xlabel("Column Index")
    ax.set_ylabel("Row Index")
    ax.grid(False)
    ax.set_facecolor('#f8f9fa')
  
  def _plot_value_distribution(self,matrix: sparse.spmatrix,ax: Axes) -> None:
    """Plot distribution of non-zero values"""
    data = matrix.data
    sns.histplot(data=data, bins=min(50, len(np.unique(data))),ax=ax, color='#2E86C1', alpha=0.7)
    ax.set_title("Value Distribution", pad=10)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
      
  def _plot_nonzero_distribution(self,matrix: sparse.spmatrix,ax: Axes) -> None:
    """Plot distribution of non-zeros per row/column"""
    row_nnz = np.diff(matrix.indptr)
    col_nnz = np.bincount(matrix.indices, minlength=matrix.shape[1])
    
    sns.kdeplot(data=row_nnz, ax=ax, label='Row NNZ', color='#2E86C1')
    sns.kdeplot(data=col_nnz, ax=ax, label='Column NNZ', color='#E74C3C')
    ax.set_title("Non-zeros Distribution", pad=10)
    ax.set_xlabel("Number of Non-zeros")
    ax.set_ylabel("Density")
    ax.legend()
  
  def _plot_block_density(self,matrix: sparse.spmatrix,ax: Axes,block_size: int = 32) -> None:
    """Plot block density heatmap"""
    rows, cols = matrix.shape
    n_blocks_row = rows // block_size
    n_blocks_col = cols // block_size
      
    density_matrix = np.zeros((n_blocks_row, n_blocks_col))
    for i in range(n_blocks_row):
      for j in range(n_blocks_col):
        block = matrix[i*block_size:(i+1)*block_size,j*block_size:(j+1)*block_size]
        density_matrix[i, j] = block.nnz / (block_size * block_size)
      
    sns.heatmap(density_matrix, ax=ax, cmap=self.config.color_map,cbar_kws={'label': 'Density'})
    ax.set_title(f"Block Density ({block_size}x{block_size})", pad=10)
    ax.set_xlabel("Block Column Index")
    ax.set_ylabel("Block Row Index")

  def _plot_value_frequencies(self, matrix: sparse.spmatrix,ax: Axes,top_k: int = 10) -> None:
    """Plot frequencies of top occurring values"""
    unique_vals, counts = np.unique(matrix.data, return_counts=True)
    sorted_indices = np.argsort(counts)[-top_k:]
    
    percentages = counts[sorted_indices] / matrix.nnz * 100
    bars = ax.bar(range(len(sorted_indices)), percentages, color='#2E86C1', alpha=0.7)
    
    ax.set_title("Top Value Frequencies", pad=10)
    ax.set_xlabel("Value Rank")
    ax.set_ylabel("% of Non-zeros")
    
    for idx, rect in enumerate(bars):
      value = unique_vals[sorted_indices[idx]]
      height = rect.get_height()
      ax.text(rect.get_x() + rect.get_width()/2., height, f'{value:.2e}', ha='center', va='bottom', rotation=45, fontsize=8)
  
  def _plot_basic_stats(self, stats: Any, ax: Axes) -> None:
    """Plot basic statistics text"""
    ax.axis('off')
    stats_text = self._format_basic_stats(stats)
    ax.text(0.05, 0.95, stats_text,transform=ax.transAxes,verticalalignment='top',fontfamily='monospace',fontsize=10)
  
  def _plot_detailed_stats(self, stats: Any, ax: Axes) -> None:
    """Plot detailed statistics text"""
    ax.axis('off')
    stats_text = self._format_detailed_stats(stats)
    ax.text(0.05, 0.95, stats_text,transform=ax.transAxes, verticalalignment='top',fontfamily='monospace',fontsize=9)
  
  def _format_basic_stats(self, stats: Dict[str, Any]) -> str:
    """Format basic statistics as string"""
    matrix_info = stats['matrix_info']
    basic_stats = stats['basic_stats']
    
    return (
      f"Matrix Properties:\n"
      f"  • Dimensions: {matrix_info['rows']:,} × {matrix_info['cols']:,}\n"
      f"  • Nonzeros: {matrix_info['nnz']:,}\n"
      f"  • Density: {matrix_info['density']:.2%}\n\n"
      f"Value Statistics:\n"
      f"  • Range: [{basic_stats['min_value']:.2e}, {basic_stats['max_value']:.2e}]\n"
      f"  • Mean: {basic_stats['mean_value']:.2e}\n"
      f"  • Std Dev: {basic_stats['std_value']:.2e}"
    )
  
  def _format_detailed_stats(self, stats: Dict[str, Any]) -> str:
    """Format detailed statistics as string"""
    matrix_info = stats['matrix_info']
    basic_stats = stats['basic_stats']
    stat_analysis = stats.get('statistical_analysis', {})
    
    return (
      f"Matrix Properties:\n"
      f"  • Dimensions: {matrix_info['rows']:,} × {matrix_info['cols']:,}\n"
      f"  • Nonzeros: {matrix_info['nnz']:,}\n"
      f"  • Density: {matrix_info['density']:.2%}\n"
      f"  • Memory Size: {matrix_info['size_mb']:.2f} MB\n\n"
      f"Value Statistics:\n"
      f"  • Range: [{basic_stats['min_value']:.2e}, {basic_stats['max_value']:.2e}]\n"
      f"  • Mean: {basic_stats['mean_value']:.2e}\n"
      f"  • Std Dev: {basic_stats['std_value']:.2e}\n"
      f"  • Unique Values: {basic_stats['unique_values']:,}\n"
      f"  • Zero Elements: {basic_stats['zero_elements']:,}"
    )


### MAIN MODULE ###

class MatrixAnalyzer:
  """Main class coordinating matrix analysis pipeline"""
  
  def __init__(self, config: AnalyzerConfig):
    """Initialize the analyzer with configuration"""
    self.config = config
    self.logger = Logger.setup("matrix_analyzer", config.log_level)
    
    self.statistical_analyzer = StatisticalAnalyzer(
        significance_level=config.significance_level
    )
    
    self.pattern_analyzer = PatternAnalyzer(
      min_block_size=config.min_block_size,
      max_block_size=config.max_block_size,
      min_frequency=config.min_pattern_frequency,
      ignore_zeros=config.ignore_zero_patterns
    )
    
    self.visualizer = MatrixVisualizer(
      VisualizationConfig(
        figure_size=config.figure_size,
        dpi=config.dpi
      )
    )
      
    if self.config.output_dir:
      self.config.output_dir.mkdir(parents=True, exist_ok=True)
  
  def analyze_matrix(self, matrix_path: Union[str, Path], save_visualizations: bool = True) -> Dict[str, Any]:
    """Analyze a single matrix file"""
    try:
      matrix = self._load_matrix(matrix_path)
      if matrix is None:
        return None
          
      results = {
        'matrix_info': self._get_matrix_info(matrix, matrix_path),
        'basic_stats': self._compute_basic_stats(matrix),
      }
      
      if self.config.enable_advanced_stats:
        results['statistical_analysis'] = self.statistical_analyzer.analyze_matrix(matrix)
      
      if self.config.analyze_patterns:
        results['pattern_analysis'] = self.pattern_analyzer.analyze_matrix(matrix)
      
      if save_visualizations:
        self._create_visualizations(matrix, results, matrix_path)
      
      self._update_metadata(matrix_path, results)
      
      return results
        
    except Exception as e:
      self.logger.error(f"Error analyzing matrix {matrix_path}: {str(e)}")
      return None
  
  def analyze_directory(self, limit: Optional[int] = None, parallel: bool = True) -> pd.DataFrame:
    """Analyze all matrices in the configured directory"""
    matrix_files = list(self.config.input_dir.glob("*.mtx"))
    if limit:
      matrix_files = matrix_files[:limit]
    
    results = []
    if parallel and len(matrix_files) > 1:
      results = self._parallel_process(matrix_files)
    else:
      results = self._sequential_process(matrix_files)
    
    return self._create_summary_dataframe(results)
  
  def _load_matrix(self, matrix_path: Union[str, Path]) -> Optional[sparse.spmatrix]:
    """Load matrix from file with error handling"""
    try:
      matrix = mmread(str(matrix_path))
      
      if np.iscomplexobj(matrix):
        matrix = abs(matrix)
      
      return matrix.tocsr() # store as CSR (???)
        
    except Exception as e:
      self.logger.error(f"Error loading matrix {matrix_path}: {str(e)}")
      return None
  
  def _get_matrix_info(self, matrix: sparse.spmatrix, matrix_path: Union[str, Path]) -> Dict[str, Any]:
    """Get basic matrix information"""
    return {
      'filename': Path(matrix_path).name,
      'rows': matrix.shape[0],
      'cols': matrix.shape[1],
      'nnz': matrix.nnz,
      'density': matrix.nnz / (matrix.shape[0] * matrix.shape[1]),
      'size_mb': self._estimate_memory_usage(matrix) / (1024 * 1024)
    }
  
  def _compute_basic_stats(self, matrix: sparse.spmatrix) -> Dict[str, Any]:
    """Compute basic matrix statistics"""
    data = matrix.data
    
    return {
      'min_value': float(np.min(data)),
      'max_value': float(np.max(data)),
      'mean_value': float(np.mean(data)),
      'std_value': float(np.std(data)),
      'unique_values': len(np.unique(data)),
      'zero_elements': matrix.shape[0] * matrix.shape[1] - matrix.nnz
    }
  
  def _estimate_memory_usage(self, matrix: sparse.spmatrix) -> int:
    """Estimate memory usage of sparse matrix in bytes"""
    return (matrix.data.nbytes + matrix.indptr.nbytes +  matrix.indices.nbytes)
  
  def _create_visualizations(self, matrix: sparse.spmatrix, results: Dict[str, Any], matrix_path: Union[str, Path]) -> None:
    """Create requested visualizations"""
    filename = Path(matrix_path).name
    
    for viz_type in self.config.visualization_types:
      try:
        self.visualizer.create_visualization(
          matrix=matrix,
          stats=results,
          filename=filename,
          output_dir=self.config.output_dir,
          viz_type=viz_type.name.lower()
        )
      except Exception as e:
        self.logger.error(f"Error creating {viz_type} visualization: {str(e)}")

  def _update_metadata(self, matrix_path: Union[str, Path], results: Dict[str, Any]) -> None:
    """Update metadata file with analysis results"""
    try:
      metadata_path = self.config.output_dir / self.config.metadata_file
      
      metadata_entry = {
        'filename': Path(matrix_path).name,
        'timestamp': pd.Timestamp.now().isoformat(),
      }
      flattened_results = self._flatten_dict(results)
      metadata_entry.update(flattened_results)
      
      if metadata_path.exists():
        df = pd.read_csv(metadata_path)
        if metadata_entry['filename'] in df['filename'].values:
          idx = df.index[df['filename'] == metadata_entry['filename']][0]
          for key, value in metadata_entry.items():
            df.at[idx, key] = value
        else:
          df = pd.concat([df, pd.DataFrame([metadata_entry])], ignore_index=True)
      else:
        df = pd.DataFrame([metadata_entry])
      
      df.to_csv(metadata_path, index=False)
      
    except Exception as e:
      self.logger.error(f"Error updating metadata: {str(e)}")
  
  def _parallel_process(self, matrix_files: List[Path]) -> List[Dict[str, Any]]:
    """Process matrices in parallel"""
    results = []
    with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
      future_to_path = {
        executor.submit(self.analyze_matrix, path): path 
        for path in matrix_files
      }
        
      with tqdm(total=len(matrix_files), desc="Processing matrices") as pbar:
        for future in future_to_path:
          try:
            result = future.result()
            if result is not None:
              results.append(result)
          except Exception as e:
            self.logger.error(f"Error processing {future_to_path[future]}: {str(e)}")
          pbar.update(1)
    
    return results
  
  def _sequential_process(self, matrix_files: List[Path]) -> List[Dict[str, Any]]:
    """Process matrices sequentially"""
    results = []
    for matrix_path in tqdm(matrix_files, desc="Processing matrices"):
      result = self.analyze_matrix(matrix_path)
      if result is not None:
        results.append(result)
    return results
  
  def _create_summary_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create summary DataFrame from analysis results"""
    if not results:
      return pd.DataFrame()
        
    flat_results = [self._flatten_dict(r) for r in results]
    
    df = pd.DataFrame(flat_results)
    
    if 'filename' in df.columns:
      df.set_index('filename', inplace=True)
    
    return df
  
  @staticmethod
  def _flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """Flatten nested dictionary with custom separator"""
    items: List = []
    for k, v in d.items():
      new_key = f"{parent_key}{sep}{k}" if parent_key else k
      
      if isinstance(v, dict):
        items.extend(MatrixAnalyzer._flatten_dict(v, new_key, sep).items())
      else:
        items.append((new_key, v))
    
    return dict(items)

def print_analysis_summary(results: Dict[str, Any]) -> None:
  """Print formatted summary of single matrix analysis"""
  matrix_info = results['matrix_info']
  basic_stats = results['basic_stats']
  
  print(f"Filename: {matrix_info['filename']}")
  print(f"Dimensions: {matrix_info['rows']} × {matrix_info['cols']}")
  print(f"Non-zeros: {matrix_info['nnz']:,}")
  print(f"Density: {matrix_info['density']:.2%}")
  print(f"Size in memory: {matrix_info['size_mb']:.2f} MB")
  print("\nValue Statistics:")
  print(f"  Range: [{basic_stats['min_value']:.2e}, {basic_stats['max_value']:.2e}]")
  print(f"  Mean: {basic_stats['mean_value']:.2e}")
  print(f"  Std Dev: {basic_stats['std_value']:.2e}")
  print(f"  Unique Values: {basic_stats['unique_values']:,}")
  
  if 'statistical_analysis' in results:
    stats = results['statistical_analysis']
    print("\nAdvanced Statistics:")
    print(f"  Symmetry Score: {stats.symmetry_score:.2%}")
    print(f"  Spatial Correlation: {stats.spatial_correlation:.2f}")
    print(f"  Computational Density: {stats.computational_density:.2f}")

  if 'pattern_analysis' in results:
    patterns = results['pattern_analysis']
    print("\nPattern Analysis:")
    print(f"  Total Patterns Found: {sum(len(p) for p in patterns.values())}")
    for size, size_patterns in patterns.items():
      print(f"  {size}x{size} Blocks: {len(size_patterns)} patterns")

def print_directory_summary(results_df: pd.DataFrame) -> None:
  """Print formatted summary of directory analysis"""
  print(f"Total Matrices Analyzed: {len(results_df)}")

  print("\nMatrix Sizes:")
  print(f"  Smallest: {results_df['matrix_info_rows'].min():,} × {results_df['matrix_info_cols'].min():,}")
  print(f"  Largest: {results_df['matrix_info_rows'].max():,} × {results_df['matrix_info_cols'].max():,}")
  print(f"  Median: {results_df['matrix_info_rows'].median():,.0f} × {results_df['matrix_info_cols'].median():,.0f}")
  
  print("\nDensity Statistics:")
  print(f"  Minimum: {results_df['matrix_info_density'].min():.2%}")
  print(f"  Maximum: {results_df['matrix_info_density'].max():.2%}")
  print(f"  Median: {results_df['matrix_info_density'].median():.2%}")
  
  print("\nValue Statistics:")
  print(f"  Global Min: {results_df['basic_stats_min_value'].min():.2e}")
  print(f"  Global Max: {results_df['basic_stats_max_value'].max():.2e}")
  print(f"  Mean Range: [{results_df['basic_stats_mean_value'].mean():.2e} ± {results_df['basic_stats_std_value'].mean():.2e}]")

  if 'pattern_analysis' in results_df.columns:
    pattern_counts = results_df['pattern_analysis'].apply( lambda x: sum(len(p) for p in x.values()) )
    print("\nPattern Statistics:")
    print(f"  Average Patterns per Matrix: {pattern_counts.mean():.1f}")
    print(f"  Most Patterns in Single Matrix: {pattern_counts.max()}")
    print(f"  Least Patterns in Single Matrix: {pattern_counts.min()}")

def print_help_examples():
  """Print helpful usage examples"""
  examples = """
Example Usage:
-------------
1. Basic analysis of a single matrix:
   python matrix_analyzer.py matrix.mtx

2. Detailed analysis with all visualizations:
   python matrix_analyzer.py matrix.mtx --visualizations all --advanced-stats --analyze-patterns

3. Process all matrices in a directory in parallel:
   python matrix_analyzer.py matrices_dir/ --parallel --workers 8

4. Custom visualization settings:
   python matrix_analyzer.py matrix.mtx --figure-size 20 15 --dpi 600

5. Save configuration for later use:
   python matrix_analyzer.py matrix.mtx --advanced-stats --save-config my_config.json

6. Use saved configuration:
   python matrix_analyzer.py matrix.mtx --config my_config.json
"""
  print(examples)

def create_parser() -> argparse.ArgumentParser:
  """Create command line argument parser"""
  parser = argparse.ArgumentParser(
    description='Enhanced Matrix Market File Visualizer',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  
  parser.add_argument('matrix_dir', type=str, help='Directory containing .mtx files and metadata')
  parser.add_argument('--limit', type=int,help='Limit the number of matrices to process')
  parser.add_argument('--detailed', action='store_true',help='Generate detailed visualizations and statistics')
  parser.add_argument('--config', type=str, help='Load configuration from JSON file')
  parser.add_argument('--save-config', type=str, help='Save configuration to JSON file')
  
  parser.add_argument('--no-plots', action='store_true',help='Skip generating visualization plots')
  parser.add_argument('--no-metadata-update', action='store_true',help='Skip updating metadata file with new statistics')
  parser.add_argument('--output-dir', type=str,help='Custom directory for visualization outputs')
  
  parser.add_argument('--block-sizes', type=int, nargs='+',default=[2, 4, 8, 16, 32],help='Block sizes for density analysis')
  parser.add_argument('--analyze-patterns', action='store_true',help='Perform detailed block pattern analysis')
  parser.add_argument('--max-block-size', type=int, default=8,help='Maximum block size for pattern analysis')
  parser.add_argument('--max-patterns', type=int, default=10,help='Maximum number of patterns to display per block size')

  parser.add_argument('--plot-size', type=float, nargs=2, default=[15, 10],help='Figure size in inches (width height)')

  parser.add_argument('--log-level',choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],default='INFO',help='Set the logging level')
  
  parser.add_argument('--parallel', action='store_true', help='Enable parallel processing')
  parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers')
  
  return parser

def process_visualization_types(types: List[str]) -> List[VisualizationType]:
  """Process visualization types from command line arguments"""
  valid_types = [v.name.lower() for v in VisualizationType]
  return [VisualizationType[t.upper()] for t in types if t.lower() in valid_types]

def main():
  """Main execution function"""
  parser = create_parser()
  args = parser.parse_args()

  try:
    config = None
    if hasattr(args, 'config') and args.config:
      config = AnalyzerConfig.from_json(args.config)
    else:
      config = AnalyzerConfig(
        input_dir=Path(args.matrix_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        parallel_workers=args.workers if args.parallel else 1,
        visualization_types=[VisualizationType.DETAILED] if args.detailed else [VisualizationType.BASIC],
        figure_size=tuple(getattr(args, 'plot_size', [15, 10])),
        enable_advanced_stats=getattr(args, 'detailed', False),
        log_level=args.log_level,
        analyze_patterns=args.analyze_patterns,
      )
    
    logger = Logger.setup("matrix_analyzer", config.log_level)
    
    analyzer = MatrixAnalyzer(config)
    input_path = Path(args.matrix_dir)
    
    if input_path.is_file():
      logger.info(f"Analyzing single matrix: {input_path}")
      results = analyzer.analyze_matrix(
        input_path,
        save_visualizations=not args.no_plots
      )
      
      if results:
        print("\nAnalysis Summary:")
        print("-" * 50)
        print_analysis_summary(results)
          
    elif input_path.is_dir():
      logger.info(f"Analyzing matrices in directory: {input_path}")
      results_df = analyzer.analyze_directory(
        limit=args.limit,
        parallel=args.parallel
      )
      
      if not results_df.empty:
        output_path = config.output_dir / "analysis_results.csv"
        results_df.to_csv(output_path)
        print(f"\nResults saved to {output_path}")
        
        print_directory_summary(results_df)

    else:
      raise FileNotFoundError(f"Input path does not exist: {input_path}")
    
    logger.info("Analysis completed successfully")
        
  except Exception as e:
    logger.error(f"Error during execution: {str(e)}")
    if args.log_level == 'DEBUG':
      logger.exception("Detailed error information:")
    sys.exit(1)

if __name__ == "__main__":
  if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help', '--examples']:
    print_help_examples()
    parser = create_parser()
    parser.print_help()
  else:
    main()
