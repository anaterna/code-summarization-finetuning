from git import Repo
from pathlib import Path

class AirflowRepoManager:
    """Class that creates an airflow manager"""
    def __init__(self, release="3.0.0", clone_dir="airflow-repository", target_subdir="airflow-core"):
        """
        Initialize the repository manager.

        Args:
            release (str): Tag or branch to checkout.
            clone_dir (str): Local directory where the repo will be cloned.
            target_subdir (str): Subdirectory within the repo to focus on.
        """
        self.repo_url = "https://github.com/apache/airflow.git"
        self.release = release
        self.clone_dir = Path(clone_dir)
        self.target_subdir = Path(target_subdir)
        self.root_dir = None  # Will hold the final path after cloning

    def clone_repo(self):
        """
        Clone the repository if it doesn't exist locally.
        """
        if not self.clone_dir.exists():
            print(f"Cloning {self.repo_url} (release {self.release}) into {self.clone_dir}...")
            Repo.clone_from(
                self.repo_url,
                self.clone_dir,
                depth=1,
                branch=self.release
            )
            print("Clone completed.")
        else:
            print(f"Repository already exists at {self.clone_dir}")

        self.root_dir = self.clone_dir / self.target_subdir
        return self.root_dir
