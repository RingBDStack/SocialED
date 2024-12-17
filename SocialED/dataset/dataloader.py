import numpy as np
import os
from git import Repo, GitCommandError
import pandas as pd
import shutil
from uuid import uuid4
from datetime import datetime
import subprocess
import tempfile

class DatasetLoader:
    def __init__(self, dataset=None, dir_path=None):
        self.dir_path = dir_path
        self.dataset = dataset
        self.default_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataset/data"))
        print(f"Data root path: {self.default_root_path}")  # 调试信息
        os.makedirs(self.default_root_path, exist_ok=True)
        
        self.required_columns = [
            'tweet_id', 'text', 'event_id', 'words', 'filtered_words',
            'entities', 'user_id', 'created_at', 'urls', 'hashtags', 'user_mentions'
        ]
        self.repo_url = "https://github.com/ChenBeici/SocialED_datasets.git"
        self.target_folder = "npy_data"

    def download_and_cleanup(self, repo_url, dataset_name, local_target_folder):
        # 创建临时目录
        local_repo_path = os.path.join(os.path.dirname(__file__), "tmp", str(uuid4()))
        try:
            print(f"Downloading {dataset_name}.npy from {repo_url}")
            
            # 克隆仓库
            subprocess.run(['git', 'clone', '--branch', 'main', repo_url, local_repo_path], check=True)
            
            # 确保目标目录存在
            os.makedirs(local_target_folder, exist_ok=True)
            print(f"Target directory: {local_target_folder}")  # 调试信息
            
            # 搜索.npy文件
            npy_files = []
            for root, dirs, files in os.walk(local_repo_path):
                for file in files:
                    if file == f'{dataset_name}.npy':
                        npy_files.append(os.path.join(root, file))
            
            if npy_files:
                target_file = os.path.join(local_target_folder, f'{dataset_name}.npy')
                print(f"Copying from {npy_files[0]} to {target_file}")  # 调试信息
                shutil.copy2(npy_files[0], target_file)
                return True
            else:
                print(f"Error: {dataset_name}.npy not found in repository")
                return False
                
        except Exception as e:
            print(f"Error during download: {str(e)}")
            return False
        finally:
            if os.path.exists(local_repo_path):
                shutil.rmtree(local_repo_path)

    def download(self):
        local_target_folder = os.path.join(self.default_root_path, self.dataset)
        return self.download_and_cleanup(
            self.repo_url,
            self.dataset,
            local_target_folder
        )

    def load_data(self):
        raise NotImplementedError("Subclasses must implement load_data()")

    def get_dataset_language(self):
        """
        Determine the language based on the current dataset.
        
        Returns:
            str: The language of the dataset ('English', 'French', 'Arabic').
        """
        dataset_language_map = {
            'MAVEN': 'English',
            'Event2012': 'English', 
            'Event2018': 'French',
            'Arabic_Twitter': 'Arabic',
            'CrisisLexT26': 'English',
            'CrisisLexT6': 'English', 
            'CrisisMMD': 'English',
            'CrisisNLP': 'English',
            'HumAID': 'English',
            'ICWSM2018': 'English',
            'ISCRAM2013': 'English',
            'BigCrisisData': 'English',
            'KBP': 'English',
            'Event2012_100': 'English',
            'Event2018_100': 'French',
            'Arabic_100': 'Arabic'
        }
        
        language = dataset_language_map.get(self.dataset)
        if not language:
            raise ValueError(f"Unsupported dataset: {self.dataset}. Supported datasets are: {', '.join(dataset_language_map.keys())}")
        return language

class MAVEN(DatasetLoader):
    def __init__(self, dir_path=None):
        super().__init__(dataset='MAVEN', dir_path=dir_path)
    
    def load_data(self):
        dataset_path = os.path.join(self.default_root_path, self.dataset)
        print(f"Dataset path: {dataset_path}")  # 调试信息
        
        if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
            print(f"Directory {dataset_path} does not exist or is empty, downloading...")
            if not self.download():
                raise RuntimeError("Failed to download dataset")
        
        file_path = os.path.join(dataset_path, f'{self.dataset}.npy')
        print(f"Loading file from: {file_path}")  # 调试信息
        
        if not os.path.exists(file_path):
            print(f"File not found at: {file_path}")
            print(f"Directory contents: {os.listdir(dataset_path) if os.path.exists(dataset_path) else 'Directory does not exist'}")
            raise FileNotFoundError(f"Data file not found at {file_path}")
        
        data = np.load(file_path, allow_pickle=True)
        df = pd.DataFrame(data, columns=self.required_columns)
        print("MAVEN dataset loaded successfully.")
        return df

class CrisisNLP(DatasetLoader):
    def __init__(self, dir_path=None):
        super().__init__(dataset='CrisisNLP', dir_path=dir_path)
    
    def load_data(self):
        dataset_path = os.path.join(self.default_root_path, self.dataset)
        print(f"Dataset path: {dataset_path}")  # 调试信息
        
        if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
            print(f"Directory {dataset_path} does not exist or is empty, downloading...")
            if not self.download():
                raise RuntimeError("Failed to download dataset")
        
        file_path = os.path.join(dataset_path, f'{self.dataset}.npy')
        print(f"Loading file from: {file_path}")  # 调试信息
        
        if not os.path.exists(file_path):
            print(f"File not found at: {file_path}")
            print(f"Directory contents: {os.listdir(dataset_path) if os.path.exists(dataset_path) else 'Directory does not exist'}")
            raise FileNotFoundError(f"Data file not found at {file_path}")
        
        data = np.load(file_path, allow_pickle=True)
        df = pd.DataFrame(data, columns=self.required_columns)
        print("CrisisNLP dataset loaded successfully.")
        return df

class Event2012(DatasetLoader):
    def __init__(self, dir_path=None):
        super().__init__(dataset='Event2012', dir_path=dir_path)
    
    def load_data(self):
        dataset_path = os.path.join(self.default_root_path, self.dataset)
        print(f"Dataset path: {dataset_path}")  # 调试信息
        
        if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
            print(f"Directory {dataset_path} does not exist or is empty, downloading...")
            if not self.download():
                raise RuntimeError("Failed to download dataset")
        
        file_path = os.path.join(dataset_path, f'{self.dataset}.npy')
        print(f"Loading file from: {file_path}")  # 调试信息
        
        if not os.path.exists(file_path):
            print(f"File not found at: {file_path}")
            print(f"Directory contents: {os.listdir(dataset_path) if os.path.exists(dataset_path) else 'Directory does not exist'}")
            raise FileNotFoundError(f"Data file not found at {file_path}")
        
        data = np.load(file_path, allow_pickle=True)
        df = pd.DataFrame(data, columns=self.required_columns)
        print("Event2012 dataset loaded successfully.")
        return df

class Event2018(DatasetLoader):
    def __init__(self, dir_path=None):
        super().__init__(dataset='Event2018', dir_path=dir_path)
    
    def load_data(self):
        dataset_path = os.path.join(self.default_root_path, self.dataset)
        print(f"Dataset path: {dataset_path}")  # 调试信息
        
        if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
            print(f"Directory {dataset_path} does not exist or is empty, downloading...")
            if not self.download():
                raise RuntimeError("Failed to download dataset")
        
        file_path = os.path.join(dataset_path, f'{self.dataset}.npy')
        print(f"Loading file from: {file_path}")  # 调试信息
        
        if not os.path.exists(file_path):
            print(f"File not found at: {file_path}")
            print(f"Directory contents: {os.listdir(dataset_path) if os.path.exists(dataset_path) else 'Directory does not exist'}")
            raise FileNotFoundError(f"Data file not found at {file_path}")
        
        data = np.load(file_path, allow_pickle=True)
        df = pd.DataFrame(data, columns=self.required_columns)
        print("Event2018 dataset loaded successfully.")
        return df

class ArabicTwitter(DatasetLoader):
    def __init__(self, dir_path=None):
        super().__init__(dataset='Arabic_Twitter', dir_path=dir_path)
    
    def load_data(self):
        dataset_path = os.path.join(self.default_root_path, self.dataset)
        print(f"Dataset path: {dataset_path}")  # 调试信息
        
        if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
            print(f"Directory {dataset_path} does not exist or is empty, downloading...")
            if not self.download():
                raise RuntimeError("Failed to download dataset")
        
        file_path = os.path.join(dataset_path, f'{self.dataset}.npy')
        print(f"Loading file from: {file_path}")  # 调试信息
        
        if not os.path.exists(file_path):
            print(f"File not found at: {file_path}")
            print(f"Directory contents: {os.listdir(dataset_path) if os.path.exists(dataset_path) else 'Directory does not exist'}")
            raise FileNotFoundError(f"Data file not found at {file_path}")
        
        data = np.load(file_path, allow_pickle=True)
        df = pd.DataFrame(data, columns=self.required_columns)
        print("Arabic Twitter dataset loaded successfully.")
        return df

class CrisisLexT26(DatasetLoader):
    def __init__(self, dir_path=None):
        super().__init__(dataset='CrisisLexT26', dir_path=dir_path)
    
    def load_data(self):
        dataset_path = os.path.join(self.default_root_path, self.dataset)
        print(f"Dataset path: {dataset_path}")  # 调试信息
        
        if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
            print(f"Directory {dataset_path} does not exist or is empty, downloading...")
            if not self.download():
                raise RuntimeError("Failed to download dataset")
        
        file_path = os.path.join(dataset_path, f'{self.dataset}.npy')
        print(f"Loading file from: {file_path}")  # 调试信息
        
        if not os.path.exists(file_path):
            print(f"File not found at: {file_path}")
            print(f"Directory contents: {os.listdir(dataset_path) if os.path.exists(dataset_path) else 'Directory does not exist'}")
            raise FileNotFoundError(f"Data file not found at {file_path}")
        
        data = np.load(file_path, allow_pickle=True)
        df = pd.DataFrame(data, columns=self.required_columns)
        print("CrisisLexT26 dataset loaded successfully.")
        return df

class CrisisMMD(DatasetLoader):
    def __init__(self, dir_path=None):
        super().__init__(dataset='CrisisMMD', dir_path=dir_path)
    
    def load_data(self):
        dataset_path = os.path.join(self.default_root_path, self.dataset)
        print(f"Dataset path: {dataset_path}")  # 调试信息
        
        if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
            print(f"Directory {dataset_path} does not exist or is empty, downloading...")
            if not self.download():
                raise RuntimeError("Failed to download dataset")
        
        file_path = os.path.join(dataset_path, f'{self.dataset}.npy')
        print(f"Loading file from: {file_path}")  # 调试信息
        
        if not os.path.exists(file_path):
            print(f"File not found at: {file_path}")
            print(f"Directory contents: {os.listdir(dataset_path) if os.path.exists(dataset_path) else 'Directory does not exist'}")
            raise FileNotFoundError(f"Data file not found at {file_path}")
        
        data = np.load(file_path, allow_pickle=True)
        df = pd.DataFrame(data, columns=self.required_columns)
        print("CrisisMMD dataset loaded successfully.")
        return df

class HumAID(DatasetLoader):
    def __init__(self, dir_path=None):
        super().__init__(dataset='HumAID', dir_path=dir_path)
    
    def load_data(self):
        dataset_path = os.path.join(self.default_root_path, self.dataset)
        print(f"Dataset path: {dataset_path}")  # 调试信息
        
        if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
            print(f"Directory {dataset_path} does not exist or is empty, downloading...")
            if not self.download():
                raise RuntimeError("Failed to download dataset")
        
        file_path = os.path.join(dataset_path, f'{self.dataset}.npy')
        print(f"Loading file from: {file_path}")  # 调试信息
        
        if not os.path.exists(file_path):
            print(f"File not found at: {file_path}")
            print(f"Directory contents: {os.listdir(dataset_path) if os.path.exists(dataset_path) else 'Directory does not exist'}")
            raise FileNotFoundError(f"Data file not found at {file_path}")
        
        data = np.load(file_path, allow_pickle=True)
        df = pd.DataFrame(data, columns=self.required_columns)
        print("HumAID dataset loaded successfully.")
        return df

class KBP(DatasetLoader):
    def __init__(self, dir_path=None):
        super().__init__(dataset='KBP', dir_path=dir_path)
    
    def load_data(self):
        dataset_path = os.path.join(self.default_root_path, self.dataset)
        print(f"Dataset path: {dataset_path}")  # 调试信息
        
        if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
            print(f"Directory {dataset_path} does not exist or is empty, downloading...")
            if not self.download():
                raise RuntimeError("Failed to download dataset")
        
        file_path = os.path.join(dataset_path, f'{self.dataset}.npy')
        print(f"Loading file from: {file_path}")  # 调试信息
        
        if not os.path.exists(file_path):
            print(f"File not found at: {file_path}")
            print(f"Directory contents: {os.listdir(dataset_path) if os.path.exists(dataset_path) else 'Directory does not exist'}")
            raise FileNotFoundError(f"Data file not found at {file_path}")
        
        data = np.load(file_path, allow_pickle=True)
        df = pd.DataFrame(data, columns=self.required_columns)
        print("KBP dataset loaded successfully.")
        return df

if __name__ == "__main__":
    # Test MAVEN dataset
    #maven = MAVEN()
    dataset = MAVEN().load_data()
    print("MAVEN dataset head:")
    print(dataset.head())
    