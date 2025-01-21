import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Union, Dict
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass
from tqdm import tqdm
import csv
import tarfile
import tempfile
import random
import sys


@dataclass
class MatrixCriteria:
  """Class to hold matrix search criteria with sensible defaults"""
  rows_range: Optional[Tuple[int, int]] = (0, float("inf")) # None # (min, max)
  cols_range: Optional[Tuple[int, int]] = (0, float("inf")) # None # (min, max)
  size_target: Optional[Tuple[int, float]] = None # (target, margin)

  sparsity_range: Optional[Tuple[float, float]] = (0.0, 1.0) # None # (min, max)
  sparsity_target: Optional[Tuple[float, float]] = None # (target, margin)
  nonzeros_range: Optional[Tuple[int, int]] = (0, float("inf")) # None # (min, max)

  matrix_types: Optional[List[str]] = None
  symmetric: Optional[bool] = None
  real: Optional[bool] = None
  include_missing: bool = False


class MatrixDownloader:
  """Main class for downloading sparse matrices"""

  BASE_URL = "https://sparse.tamu.edu/"

  def __init__(self,
    criteria: Optional[MatrixCriteria] = None,
    download_dir: Optional[str] = None,
    log_level: int = logging.INFO,
  ):
    self.criteria = criteria or MatrixCriteria()
    self.download_dir = Path(download_dir) if download_dir else Path("./matrices")
    self._setup_logging(log_level)
    self.session = requests.Session()
    self.metadata_file = None

  def _setup_logging(self, log_level: int) -> None:
    """Configure logging"""
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
    self.logger = logging.getLogger(__name__)

  def _fetch_matrix_list(self) -> pd.DataFrame:
    """Fetch and parse the matrix list from the website"""
    try:
      self.logger.info("Fetching matrix list from SuiteSparse Collection...")
      response = self.session.get(f"{self.BASE_URL}?per_page=All")
      response.raise_for_status()

      soup = BeautifulSoup(response.text, "html.parser")
      table = soup.find("table", {"id": "matrices"})

      matrices = []
      for row in table.find_all("tr")[1:]:
        cols = row.find_all("td")
        if len(cols) >= 8:
          matrix = {
            "id": int(cols[0].text.strip()),
            "name": cols[1].text.strip(),
            "group": cols[2].text.strip(),
            "rows": self._parse_number(cols[3].text.strip()),
            "cols": self._parse_number(cols[4].text.strip()),
            "nonzeros": self._parse_number(cols[5].text.strip()),
            "kind": cols[6].text.strip(),
            "download_url": self._extract_download_url(cols[8]),
          }
          matrix["sparsity"] = self._calculate_sparsity(matrix)
          matrices.append(matrix)

      df = pd.DataFrame(matrices)
      print(f"\nDEBUG: Fetched {len(df)} matrices")
      print("DEBUG: Sample of matrix types found:")
      print(df["kind"].value_counts().head())
      return df

    except Exception as e:
      self.logger.error(f"Error fetching matrix list: {str(e)}")
      raise

  @staticmethod
  def _parse_number(text: str) -> Optional[int]:
    """Parse number from text, handling commas and missing values"""
    try:
      return int(text.replace(",", "")) if text != "?" else None
    except ValueError:
      return None

  @staticmethod
  def _calculate_sparsity(matrix: Dict) -> Optional[float]:
    """Calculate sparsity ratio for a matrix"""
    try:
      if all(matrix[k] is not None for k in ["rows", "cols", "nonzeros"]):
        return matrix["nonzeros"] / (matrix["rows"] * matrix["cols"])
      return None
    except ZeroDivisionError:
      return None

  def _validate_download_url(self, url: Optional[str]) -> Optional[str]:
    """Validate and normalize the download URL"""
    if not url:
      return None
        
    if not url.endswith('.tar.gz'):
      self.logger.warning(f"Invalid Matrix Market URL format: {url}")
      return None
        
    if '/MM/' not in url:
      self.logger.warning(f"URL may not be a Matrix Market file: {url}")
      return None
        
    return url

  def _extract_download_url(self, cell) -> Optional[str]:
    """Extract Matrix Market format download URL from cell"""
    try:
      links = cell.find_all("a")
      for link in links:
        if "Matrix Market" in link.text:
          return self._validate_download_url(link["href"])
      return None
    except (AttributeError, KeyError) as e:
      self.logger.debug(f"Error extracting URL: {str(e)}")
      return None

  def _filter_matrices(self, df: pd.DataFrame) -> pd.DataFrame:
    """Apply filtering criteria to matrix DataFrame"""
    print(f"\nDEBUG: Starting filtering with {len(df)} matrices")
    print(f"DEBUG: Current criteria: {self.criteria.__dict__}")
    if not self.criteria.include_missing:
      df = df.dropna(subset=['rows', 'cols', 'nonzeros'])
      print(f"DEBUG: After dropping NA: {len(df)} matrices")

    if self.criteria.matrix_types:
      print(f"DEBUG: Filtering by matrix types: {self.criteria.matrix_types}")
      print(f"DEBUG: Available types in dataset: {sorted(df['kind'].unique())}")
      df = df[df["kind"].isin(self.criteria.matrix_types)]
      print(f"DEBUG: After type filtering: {len(df)} matrices")

    if self.criteria.size_target:
      target, margin = self.criteria.size_target
      margin_val = target * margin
      df = df[(df["rows"].between(target - margin_val, target + margin_val)) & (df["cols"].between(target - margin_val, target + margin_val))]
    elif self.criteria.rows_range:
      df = df[df["rows"].between(*self.criteria.rows_range)]
      if self.criteria.cols_range:
        df = df[df["cols"].between(*self.criteria.cols_range)]

    if self.criteria.sparsity_target:
      target, margin = self.criteria.sparsity_target
      df = df[df["sparsity"].between(target - margin, target + margin)]
    elif self.criteria.sparsity_range:
      df = df[df["sparsity"].between(*self.criteria.sparsity_range)]
    elif self.criteria.nonzeros_range:
      df = df[df["nonzeros"].between(*self.criteria.nonzeros_range)]

    print(f"DEBUG: Final filtered count: {len(df)} matrices")
    if len(df) == 0:
      print("\nDEBUG: Sample of original data types:")
      print(df["kind"].head(10))
    
    return df

  @staticmethod
  def create_from_args(args: argparse.Namespace) -> "MatrixDownloader":
    """Create MatrixDownloader instance from command line arguments"""
    criteria = MatrixCriteria()
    if args.size_target:
      criteria.size_target = (args.size_target, args.size_margin)
      criteria.rows_range = None
      criteria.cols_range = None
    elif any([args.rows_min, args.rows_max, args.cols_min, args.cols_max]):
      criteria.rows_range = (args.rows_min if args.rows_min is not None else 0, args.rows_max if args.rows_max is not None else float("inf"))
      criteria.cols_range = (args.cols_min if args.cols_min is not None else 0, args.cols_max if args.cols_max is not None else float("inf"))

    if args.sparsity_target is not None:
      criteria.sparsity_target = (args.sparsity_target, args.sparsity_margin)
      criteria.sparsity_range = None
      criteria.nonzeros_range = None
    elif args.sparsity_min is not None or args.sparsity_max is not None:
      criteria.sparsity_range = (args.sparsity_min if args.sparsity_min is not None else 0.0, args.sparsity_max if args.sparsity_max is not None else 1.0)
      criteria.nonzeros_range = None
    elif args.nonzeros_min is not None or args.nonzeros_max is not None:
      criteria.nonzeros_range = (args.nonzeros_min if args.nonzeros_min is not None else 0,args.nonzeros_max if args.nonzeros_max is not None else float("inf"))

    if args.types:
      criteria.matrix_types = args.types
      print(f"DEBUG: Matrix types set to: {criteria.matrix_types}")
    
    if args.include_missing is not None:
      criteria.include_missing = args.include_missing

    print(f"DEBUG: Final criteria settings: {criteria.__dict__}")
    return MatrixDownloader(criteria=criteria, download_dir=args.download_dir)

  def get_matching_matrices(self) -> pd.DataFrame:
    """Get DataFrame of matrices matching criteria"""
    df = self._fetch_matrix_list()
    self.logger.info(f"Initial matrix head: {df.head()}")
    return self._filter_matrices(df)

  def download_matrices(self,
    mode: str = "prompt", 
    count: int = 1, 
    directory: Optional[str] = None,
  ) -> None:
    """Download matrices based on specified mode"""
    df = self.get_matching_matrices()
    if df.empty:
      self.logger.warning("No matrices match the specified criteria")
      return

    download_dir = Path(directory) if directory else self.download_dir
    download_dir.mkdir(parents=True, exist_ok=True)

    if mode == "all":
      matrices_to_download = df
    elif mode == "sample":
      matrices_to_download = df.sample(n=min(count, len(df)))
    else:  # deafult (prompt)
      matrices_to_download = self._prompt_for_matrices(df)

    with tqdm(total=len(matrices_to_download), desc="Overall progress", unit="matrix") as pbar:
      for _, matrix in matrices_to_download.iterrows():
        try:
          self._download_matrix(matrix, download_dir)
        except Exception as e:
          self.logger.error(f"Failed to download {matrix['name']}: {str(e)}")
        pbar.update(1)

#     for _, matrix in matrices_to_download.iterrows():
#       self._download_matrix(matrix, download_dir)

  def _prompt_for_matrices(self, df: pd.DataFrame) -> pd.DataFrame:
    """Prompt user for which matrices to download"""
    selected_indices = []
    for idx, matrix in df.iterrows():
      response = input(
        f"\nDownload {matrix['name']} "
        f"({matrix['rows']}x{matrix['cols']}, "
        f"nnz: {matrix['nonzeros']}, "
        f"type: {matrix['kind']})? [y/n/q]: "
      ).lower()

      if response == "q":
        break
      elif response == "y":
        selected_indices.append(idx)

    return df.loc[selected_indices]
  
  def _save_matrix_metadata(self, matrix: pd.Series, matrix_file: Path) -> None:
    """Save matrix metadata to CSV file"""
    if self.metadata_file is None:
      self.metadata_file = self.download_dir / "matrix_metadata.csv"
        
    metadata = {
      'filename': matrix_file.name,
      'id': matrix['id'],
      'rows': matrix['rows'],
      'cols': matrix['cols'],
      'nonzeros': matrix['nonzeros'],
      'kind': matrix['kind'],
      'group': matrix['group'],
      'sparsity': matrix['sparsity']
    }
    
    file_exists = self.metadata_file.exists()
    
    with open(self.metadata_file, 'a', newline='') as f:
      writer = csv.DictWriter(f, fieldnames=metadata.keys())
      if not file_exists:
        writer.writeheader()
      writer.writerow(metadata)


  def _download_matrix(self, matrix: pd.Series, directory: Path) -> None:
    """Download and extract a single matrix"""
    if not matrix["download_url"]:
      self.logger.warning(f"No download URL for matrix {matrix['name']}")
      return

    try:
      self.logger.info(f"Downloading {matrix['name']}...")
      response = self.session.get(matrix["download_url"], stream=True)
      response.raise_for_status()
      
      total_size = int(response.headers.get('content-length', 0))
      
      with tempfile.NamedTemporaryFile(suffix='.tar.gz') as tmp_file:
        with tqdm(
          total=total_size,
          unit='iB',
          unit_scale=True,
          desc=matrix['name'],
          bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
        ) as pbar:
          for chunk in response.iter_content(chunk_size=8192):
            if chunk:
              size = tmp_file.write(chunk)
              pbar.update(size)
        
        tmp_file.flush()
        
        with tarfile.open(tmp_file.name, 'r:gz') as tar:
          mtx_file = None
          for member in tar.getmembers():
            if member.name.endswith('.mtx'):
              mtx_file = member
              break
          
          if mtx_file is None:
            raise ValueError(f"No .mtx file found in archive for {matrix['name']}")
          
          mtx_file.name = f"{matrix['name']}.mtx"
          tar.extract(mtx_file, path=directory)
          
          self.logger.info(f"Successfully extracted {matrix['name']}.mtx")
          
          self._save_matrix_metadata(matrix, directory / f"{matrix['name']}.mtx")

    except Exception as e:
      self.logger.error(f"Error downloading/extracting {matrix['name']}: {str(e)}")
      raise

def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser with mutually exclusive groups"""
    parser = argparse.ArgumentParser(
      description="Download matrices from SuiteSparse Matrix Collection",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    size_group = parser.add_mutually_exclusive_group()
    size_group.add_argument("--size-target", type=int, help="Target matrix size (rows/cols)")
    size_group.add_argument("--rows-min", type=int, help="Minimum number of rows")

    size_margin_group = parser.add_argument_group("Size margin parameters")
    size_margin_group.add_argument("--size-margin",type=float,default=0.1,help="Margin around target size (as fraction)",)
    size_margin_group.add_argument("--rows-max", type=int, help="Maximum number of rows")
    size_margin_group.add_argument("--cols-min", type=int, help="Minimum number of columns")
    size_margin_group.add_argument("--cols-max", type=int, help="Maximum number of columns")

    sparsity_group = parser.add_mutually_exclusive_group()
    sparsity_group.add_argument("--sparsity-target", type=float, help="Target sparsity ratio")
    sparsity_group.add_argument("--sparsity-min", type=float, help="Minimum sparsity ratio")
    sparsity_group.add_argument("--nonzeros-min", type=int, help="Minimum number of nonzeros")

    sparsity_margin_group = parser.add_argument_group("Sparsity margin parameters")
    sparsity_margin_group.add_argument("--sparsity-margin",type=float,default=0.01,help="Margin around target sparsity")
    sparsity_margin_group.add_argument("--sparsity-max", type=float, help="Maximum sparsity ratio")
    sparsity_margin_group.add_argument("--nonzeros-max", type=int, help="Maximum number of nonzeros")

    props_group = parser.add_argument_group("Matrix properties")
    props_group.add_argument("--types", nargs="+", help="Matrix types to include")
    props_group.add_argument("--types-file", type=str, help="File containing matrix types")
    props_group.add_argument("--symmetric", choices=["yes", "no"], help="Filter by symmetry")
    props_group.add_argument("--real", choices=["yes", "no"], help="Filter by real/complex")
    props_group.add_argument("--include-missing",action="store_true",help="Include matrices with missing data")

    output_group = parser.add_argument_group("Output options")
    output_group.add_argument("--log-level",choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],default="INFO",help="Logging level")
    output_group.add_argument("--output-csv", type=str, help="Save results to CSV")
    output_group.add_argument("--download-mode",choices=["all", "prompt", "sample"],default="prompt",help="Download mode")
    output_group.add_argument("--download-count",type=int,default=1,help="Number of matrices to download in sample mode")
    output_group.add_argument("--download-dir",type=str,default="./matrices",help="Directory to save downloaded matrices")
    output_group.add_argument("--format",choices=["matrix-market", "matlab"],default="matrix-market",help="Download format")

    return parser

def main():
  parser = create_parser()
  args = parser.parse_args()

  try:
    downloader = MatrixDownloader.create_from_args(args)
    df = downloader.get_matching_matrices()

    if args.output_csv:
      df.to_csv(args.output_csv, index=False)
      print(f"Results saved to {args.output_csv}")
    else:
      print("\nMatching matrices:")
      print(df.to_string())

    if args.download_mode:
      downloader.download_matrices(
        mode=args.download_mode,
        count=args.download_count,
        directory=args.download_dir,
      )

  except Exception as e:
    print(f"Error: {str(e)}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
  main()
