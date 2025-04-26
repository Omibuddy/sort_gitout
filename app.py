import streamlit as st
import requests
import google.generativeai as genai
import os
import subprocess
import tempfile
import shutil
from datetime import datetime
import glob
import re
import hashlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from concurrent.futures import ThreadPoolExecutor

# ---- CONFIG & AUTH ----
GEMINI_MODEL = "gemini-2.0-flash"
GITHUB_API_BASE = "https://api.github.com/repos"

# Load your Gemini key from Streamlit secrets
if 'GEMINI_API_KEY' in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel(GEMINI_MODEL)
else:
    st.warning("Gemini API key not configured. Analysis features will not work.")

# ---- SESSION STATE FOR SEARCH HISTORY AND REPO DATA ----
if "search_history" not in st.session_state:
    st.session_state.search_history = []  # [(query, timestamp)]
if "search_results" not in st.session_state:
    st.session_state.search_results = []  # Store search results for LLM
if "repo_path" not in st.session_state:
    st.session_state.repo_path = None  # Local path to cloned repo
if "branches" not in st.session_state:
    st.session_state.branches = []  # Available branches
if "tags" not in st.session_state:
    st.session_state.tags = []  # Available tags
if "current_repo_url" not in st.session_state:
    st.session_state.current_repo_url = None  # Currently cloned repo URL
if "total_commits" not in st.session_state:
    st.session_state.total_commits = 0
if "repo_info" not in st.session_state:
    st.session_state.repo_info = None

# ---- HELPERS ----

def get_repo_hash(repo_url):
    """Generate a unique hash for repo URL to use as identifier."""
    return hashlib.md5(repo_url.encode()).hexdigest()

def get_repo_url_from_path(repo_path):
    """Extract repository URL from the local path."""
    try:
        cmd = ["git", "-C", repo_path, "config", "--get", "remote.origin.url"]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        url = result.stdout.strip()
        
        # Convert SSH URLs to HTTPS URLs for linking
        if url.startswith("git@github.com:"):
            url = "https://github.com/" + url[15:].replace(".git", "")
        elif url.endswith(".git"):
            url = url[:-4]
            
        return url
    except Exception:
        # If we can't get the URL, use the one from session state
        return st.session_state.current_repo_url

def clone_repository(repo_url, token=None):
    """Clone a GitHub repository to a temporary directory if not already cloned."""
    # Check if we already have this repo cloned
    if st.session_state.repo_path and st.session_state.current_repo_url == repo_url:
        # If the repo path exists and is valid, reuse it
        if os.path.exists(st.session_state.repo_path) and os.path.isdir(os.path.join(st.session_state.repo_path, '.git')):
            # Pull latest changes instead of cloning again
            try:
                with st.status("Updating repository..."):
                    subprocess.run(
                        ["git", "-C", st.session_state.repo_path, "pull"],
                        check=True,
                        capture_output=True
                    )
                    # Refresh branches and tags
                    branches = get_local_branches(st.session_state.repo_path)
                    tags = get_local_tags(st.session_state.repo_path)
                    
                    # Get commit count for status
                    total_commits = get_total_commit_count(st.session_state.repo_path)
                    st.session_state.total_commits = total_commits
                    
                return st.session_state.repo_path, branches, tags
            except Exception as e:
                st.warning(f"Could not update repository: {str(e)}. Will try to clone again.")
                # If update fails, continue with new clone
                cleanup_repo()
    
    # Create a temporary directory for the repo
    repo_hash = get_repo_hash(repo_url)
    temp_dir = os.path.join(tempfile.gettempdir(), f"streamlit_repo_{repo_hash}")
    
    # Clean up any existing directory with the same name
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    try:
        # Add token to URL if provided
        if token:
            # Parse the repo URL to insert token
            protocol, _, domain_path = repo_url.partition("://")
            auth_url = f"{protocol}://x-access-token:{token}@{domain_path}"
        else:
            auth_url = repo_url
        
        # Clone the repository with progress indication
        with st.status("Cloning repository (this may take a while for large repositories)...") as status:
            # First try with depth 1 for speed
            try:
                subprocess.run(
                    ["git", "clone", "--depth", "1", auth_url, temp_dir],
                    check=True,
                    capture_output=True,
                    timeout=180  # 3 minutes timeout
                )
                status.update(label="Fetching all commits...")
                # Then fetch all commits
                subprocess.run(
                    ["git", "-C", temp_dir, "fetch", "--unshallow"],
                    check=True,
                    capture_output=True,
                    timeout=300  # 5 minutes timeout
                )
            except subprocess.TimeoutExpired:
                # If timeout, try without --depth for more reliable but slower clone
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
                status.update(label="Initial clone timed out. Trying full clone...")
                subprocess.run(
                    ["git", "clone", auth_url, temp_dir],
                    check=True,
                    capture_output=True
                )
            
            # Get branches and tags after cloning
            status.update(label="Getting branches and tags...")
            branches = get_local_branches(temp_dir)
            tags = get_local_tags(temp_dir)
            
            # Get total commit count for status
            status.update(label="Counting commits...")
            total_commits = get_total_commit_count(temp_dir)
            st.session_state.total_commits = total_commits
            
            status.update(label=f"Repository ready with {total_commits} commits!")
        
        # Store the current repo URL
        st.session_state.current_repo_url = repo_url
        
        return temp_dir, branches, tags
    except Exception as e:
        # Clean up if there's an error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise Exception(f"Failed to clone repository: {str(e)}")

def get_total_commit_count(repo_path):
    """Get the total count of commits in the repository."""
    try:
        result = subprocess.run(
            ["git", "-C", repo_path, "rev-list", "--count", "--all"],
            check=True,
            capture_output=True,
            text=True
        )
        return int(result.stdout.strip())
    except Exception:
        return 0

def get_local_branches(repo_path):
    """Get all available branches in the local repo."""
    try:
        # First update the remote tracking branches
        subprocess.run(
            ["git", "-C", repo_path, "fetch", "--all"],
            check=False,  # Don't fail if this doesn't work
            capture_output=True
        )
        
        result = subprocess.run(
            ["git", "-C", repo_path, "branch", "-a"],
            check=True,
            capture_output=True,
            text=True
        )
        branches = []
        for line in result.stdout.split('\n'):
            line = line.strip()
            if line.startswith('remotes/origin/'):
                branch_name = line.replace('remotes/origin/', '')
                if branch_name != 'HEAD' and not branch_name.startswith('HEAD ->'):
                    branches.append(branch_name)
        
        # Deduplicate and sort
        return sorted(list(set(branches)))
    except Exception as e:
        st.warning(f"Error getting branches: {str(e)}")
        return ["main"]  # Default to main if error

def get_local_tags(repo_path):
    """Get all tags in the local repo."""
    try:
        result = subprocess.run(
            ["git", "-C", repo_path, "tag"],
            check=True,
            capture_output=True,
            text=True
        )
        tags = [tag.strip() for tag in result.stdout.split('\n') if tag.strip()]
        return sorted(tags)
    except Exception as e:
        st.warning(f"Error getting tags: {str(e)}")
        return []

def get_github_tags(repo_url, token=None):
    """Get tags from GitHub API."""
    try:
        owner, repo = repo_url.rstrip("/").split("/")[-2:]
        
        # Remove .git extension if present
        if repo.endswith('.git'):
            repo = repo[:-4]
            
        headers = {"Authorization": f"token {token}"} if token else {}
        try:
            res = requests.get(f"{GITHUB_API_BASE}/{owner}/{repo}/tags", headers=headers)
            res.raise_for_status()
            return [tag["name"] for tag in res.json()]
        except Exception as e:
            st.warning(f"Failed to fetch tags from GitHub API: {str(e)}. Using local tags instead.")
            return []
    except Exception:
        return []

def checkout_branch_or_tag(repo_path, ref):
    """Checkout a specific branch or tag."""
    try:
        subprocess.run(
            ["git", "-C", repo_path, "checkout", ref],
            check=True,
            capture_output=True
        )
        return True
    except Exception as e:
        st.warning(f"Failed to checkout {ref}: {str(e)}")
        return False

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_repo_info(repo_url: str, token: str):
    """Get basic repository information."""
    try:
        owner, repo = repo_url.rstrip("/").split("/")[-2:]
        
        # Remove .git extension if present
        if repo.endswith('.git'):
            repo = repo[:-4]
            
        headers = {"Authorization": f"token {token}"} if token else {}
        res = requests.get(f"{GITHUB_API_BASE}/{owner}/{repo}", headers=headers)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        st.warning(f"Failed to get repository info: {str(e)}")
        return {
            "name": repo_url.split("/")[-1],
            "owner": {"login": "Unknown"},
            "stargazers_count": 0,
            "forks_count": 0
        }

def get_features_from_local_repo(repo_path):
    """Extract potential features from top-level directories in local repo."""
    features = []
    
    try:
        # Get top-level directories
        for item in os.listdir(repo_path):
            if os.path.isdir(os.path.join(repo_path, item)) and not item.startswith('.'):
                features.append(item)
        
        # Check subdirectories of common source folders
        common_src_dirs = ['src', 'app', 'lib', 'modules', 'components']
        for src_dir in common_src_dirs:
            src_path = os.path.join(repo_path, src_dir)
            if os.path.isdir(src_path):
                for subdir in os.listdir(src_path):
                    subdir_path = os.path.join(src_path, subdir)
                    if os.path.isdir(subdir_path) and not subdir.startswith('.'):
                        features.append(f"{src_dir}/{subdir}")
        
        # Sort alphabetically
        return sorted(features)
    except Exception as e:
        st.warning(f"Error getting features: {str(e)}")
        return []

def search_local_commits(repo_path, query, feature_path=None, branch=None, 
                       tag=None, sha=None, date_range=None, batch_size=1000):
    """Search commits in the local repository with various filters."""
    
    all_commits = []
    
    with st.status("Searching commits...") as status:
        # Checkout the specified branch or tag if provided
        if branch and branch != "All Branches":
            status.update(label=f"Checking out branch: {branch}...")
            checkout_branch_or_tag(repo_path, branch)
        elif tag and tag != "All Tags":
            status.update(label=f"Checking out tag: {tag}...")
            checkout_branch_or_tag(repo_path, tag)
            
        # If searching for specific commit SHA
        if sha:
            status.update(label=f"Looking for commit: {sha}...")
            try:
                # Get single commit details
                git_cmd = ["git", "-C", repo_path, "show", "--format=%H|%an|%ad|%s", "--date=short", sha]
                
                if feature_path and feature_path != "All Features":
                    git_cmd.append("--")
                    git_cmd.append(feature_path)
                    
                result = subprocess.run(git_cmd, check=True, capture_output=True, text=True)
                
                if result.stdout.strip():
                    lines = result.stdout.strip().split('\n')
                    if lines and '|' in lines[0]:
                        parts = lines[0].split('|', 3)
                        if len(parts) >= 4:
                            sha, author, date, message = parts
                            commit = {
                                "sha": sha,
                                "author": author,
                                "date": date,
                                "raw_date": date,
                                "message": message,
                                "feature_path": feature_path if feature_path else "all",
                            }
                            detailed_info = get_commit_details(repo_path, sha)
                            commit.update(detailed_info)
                            commit["url"] = f"{get_repo_url_from_path(repo_path)}/commit/{sha}"
                            all_commits.append(commit)
                
                status.update(label=f"Found {len(all_commits)} commits")
                return all_commits
            except Exception as e:
                status.update(label=f"Error searching for SHA: {str(e)}")
                return []
        
        # Prepare git log command for general search
        status.update(label="Building search query...")
        
        # Build the git log command with pagination for large repos
        base_cmd = ["git", "-C", repo_path, "log", "--format=%H|%an|%ad|%s", "--date=short"]
        
        # Add search query if provided
        if query:
            base_cmd.extend(["-i", "--grep", query])
        
        # Add feature path filter
        if feature_path and feature_path != "All Features":
            base_cmd.append("--")
            base_cmd.append(feature_path)
        
        # Get total number of commits matching criteria
        count_cmd = base_cmd.copy()
        count_cmd.append("--count")
        
        try:
            result = subprocess.run(count_cmd, check=True, capture_output=True, text=True)
            total_matching = int(result.stdout.strip() or "0")
            status.update(label=f"Found {total_matching} matching commits. Processing...")
        except Exception:
            total_matching = "many"
            status.update(label=f"Processing commits in batches...")
        
        # Process commits in batches to avoid memory issues
        offset = 0
        has_more = True
        
        while has_more:
            cmd = base_cmd.copy()
            cmd.extend(["-n", str(batch_size), "--skip", str(offset)])
            
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                lines = result.stdout.strip().split('\n')
                
                # Filter empty lines
                lines = [line for line in lines if line.strip()]
                
                if not lines:
                    has_more = False
                    continue
                
                status.update(label=f"Processing batch {offset//batch_size + 1}... ({len(all_commits)} commits so far)")
                
                # Process current batch
                batch_commits = []
                for line in lines:
                    if not line.strip():
                        continue
                        
                    parts = line.split('|', 3)
                    if len(parts) < 4:
                        continue
                        
                    sha, author, date, message = parts
                    
                    # Basic commit data
                    commit = {
                        "sha": sha,
                        "author": author,
                        "date": date,
                        "raw_date": date,
                        "message": message,
                        "feature_path": feature_path if feature_path else "all",
                    }
                    
                    # Check date range if provided
                    if date_range and len(date_range) == 2:
                        commit_date = datetime.strptime(date, "%Y-%m-%d").date()
                        if not (date_range[0] <= commit_date <= date_range[1]):
                            continue
                    
                    batch_commits.append(commit)
                
                # Get detailed info for each commit in the batch
                with ThreadPoolExecutor(max_workers=10) as executor:
                    def get_details(commit):
                        sha = commit["sha"]
                        detailed_info = get_commit_details(repo_path, sha)
                        commit.update(detailed_info)
                        commit["url"] = f"{get_repo_url_from_path(repo_path)}/commit/{sha}"
                        # Determine status based on commit message
                        status = "success"
                        status_keywords = ["fix", "bug", "error", "fail", "issue", "patch", "resolve"]
                        if any(keyword in commit['message'].lower() for keyword in status_keywords):
                            status = "failure"
                        commit["status"] = status
                        return commit
                    
                    # Process commits in parallel for speed
                    detailed_commits = list(executor.map(get_details, batch_commits))
                    all_commits.extend(detailed_commits)
                
                # Update progress
                percentage = min(100, int(len(all_commits) / total_matching * 100)) if isinstance(total_matching, int) and total_matching > 0 else "unknown"
                status.update(label=f"Processed {len(all_commits)} commits ({percentage}% complete)")
                
                # Move to next batch
                offset += batch_size
                
                # Check if we have more commits to process
                has_more = len(lines) == batch_size
                
                # Safety limit - stop after processing 50,000 commits
                if len(all_commits) >= 50000:
                    status.update(label=f"Reached maximum limit of 50,000 commits. Consider refining your search.")
                    break
                    
            except Exception as e:
                status.update(label=f"Error processing commits: {str(e)}")
                has_more = False
        
        status.update(label=f"Found {len(all_commits)} commits matching your criteria")
        
    return all_commits

def get_commit_details(repo_path, sha):
    """Get detailed information about a specific commit."""
    try:
        # Get files changed count
        stat_cmd = ["git", "-C", repo_path, "show", "--shortstat", "--format=", sha]
        stat_result = subprocess.run(stat_cmd, check=True, capture_output=True, text=True)
        
        # Parse stats
        stat_output = stat_result.stdout.strip()
        files_changed = 0
        additions = 0
        deletions = 0
        
        files_match = re.search(r'(\d+) file', stat_output)
        if files_match:
            files_changed = int(files_match.group(1))
            
        additions_match = re.search(r'(\d+) insertion', stat_output)
        if additions_match:
            additions = int(additions_match.group(1))
            
        deletions_match = re.search(r'(\d+) deletion', stat_output)
        if deletions_match:
            deletions = int(deletions_match.group(1))
        
        # Get file paths
        diff_cmd = ["git", "-C", repo_path, "diff-tree", "--no-commit-id", "--name-only", "-r", sha]
        diff_result = subprocess.run(diff_cmd, check=True, capture_output=True, text=True)
        file_paths = [path.strip() for path in diff_result.stdout.split('\n') if path.strip()]
        
        # Get file statuses (added, modified, deleted)
        changed_files_cmd = ["git", "-C", repo_path, "show", "--name-status", "--format=", sha]
        changed_files_result = subprocess.run(changed_files_cmd, check=True, capture_output=True, text=True)
        file_statuses = []
        
        for line in changed_files_result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            parts = line.split(None, 1)
            if len(parts) >= 2:
                status_code, file_path = parts
                status_map = {
                    'A': 'Added',
                    'M': 'Modified',
                    'D': 'Deleted',
                    'R': 'Renamed',
                    'C': 'Copied'
                }
                status = status_map.get(status_code[0], 'Changed')
                file_statuses.append(f"{status}: {file_path}")
        
        return {
            "files_changed": files_changed,
            "additions": additions,
            "deletions": deletions,
            "file_paths": file_paths[:10],  # Limit to 10 files for UI
            "file_statuses": file_statuses[:10],
            "context": f"{files_changed} files changed, {additions} additions, {deletions} deletions"
        }
    except Exception as e:
        return {
            "files_changed": 0,
            "additions": 0,
            "deletions": 0,
            "file_paths": [],
            "context": f"Error getting details: {str(e)}",
            "file_statuses": []
        }

def generate_commit_summary(commit):
    """Generate a plain text summary for a commit."""
    summary_lines = []
    summary_lines.append(f"This commit {'fixed an issue' if commit['status'] == 'failure' else 'added features'} in the {commit['feature_path']} area.")
    summary_lines.append(f"It changed {commit['files_changed']} files, with {commit['additions']} additions and {commit['deletions']} deletions.")
    
    # Categorize the commit
    if commit['status'] == 'failure':
        summary_lines.append("This appears to be a bugfix or issue resolution.")
    elif commit['additions'] > commit['deletions'] * 3:
        summary_lines.append("This commit primarily added new code, suggesting new functionality.")
    elif commit['deletions'] > commit['additions'] * 3:
        summary_lines.append("This commit removed substantial code, suggesting cleanup or refactoring.")
    else:
        summary_lines.append("This commit balanced additions and deletions, suggesting code refinement.")
    
    # Suggest impact based on files changed
    if any('test' in file.lower() for file in commit['file_paths']):
        summary_lines.append("Test files were modified, suggesting improved test coverage.")
    if any('doc' in file.lower() for file in commit['file_paths']):
        summary_lines.append("Documentation was updated.")
    if any(file.endswith(('.json', '.yaml', '.yml', '.xml')) for file in commit['file_paths']):
        summary_lines.append("Configuration files were changed.")
    
    return "\n".join(summary_lines)

def create_visualization(commits, chart_type="summary"):
    """Create visualizations from commit data."""
    if not commits:
        return None
    
    # Convert to DataFrame for easier manipulation
    data = []
    for commit in commits:
        status = "Feature/Enhancement" if commit.get('status', 'success') == "success" else "Fix/Error"
        data.append({
            'date': commit['date'],
            'files_changed': commit.get('files_changed', 0),
            'additions': commit.get('additions', 0),
            'deletions': commit.get('deletions', 0),
            'author': commit['author'],
            'status': status
        })
    
    df = pd.DataFrame(data)
    
    # Skip if no data
    if df.empty:
        return None
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Summary chart shows multiple metrics
    if chart_type == "summary":
        # Create a figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Commits by status
        status_counts = df['status'].value_counts()
        if not status_counts.empty:
            axs[0, 0].pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', 
                          colors=['#4CAF50', '#F44336'] if 'Fix/Error' in status_counts.index else ['#4CAF50'])
            axs[0, 0].set_title('Commits by Type')
        
        # Plot 2: Files changed over time
        if not df.empty and 'date' in df.columns and 'files_changed' in df.columns:
            df_sorted = df.sort_values('date')
            axs[0, 1].plot(df_sorted['date'], df_sorted['files_changed'], marker='o', markersize=3, linewidth=1, label='Files')
            axs[0, 1].set_title('Files Changed Over Time')
            axs[0, 1].set_ylabel('Count')
            axs[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Additions vs Deletions by status
        if not df.empty and 'status' in df.columns and 'additions' in df.columns and 'deletions' in df.columns:
            additions_by_status = df.groupby('status')['additions'].sum()
            deletions_by_status = df.groupby('status')['deletions'].sum()
            
            status_labels = additions_by_status.index
            axs[1, 0].bar(status_labels, additions_by_status, label='Additions', alpha=0.7, color='green')
            axs[1, 0].bar(status_labels, deletions_by_status, bottom=additions_by_status, label='Deletions', alpha=0.7, color='red')
            axs[1, 0].set_title('Additions vs Deletions by Type')
            axs[1, 0].set_ylabel('Lines of Code')
            axs[1, 0].legend()
        
        # Plot 4: Commits by author (top 10)
        if not df.empty and 'author' in df.columns:
            author_counts = df['author'].value_counts().head(10)  # Top 10 contributors
            
            # Handle the case when there are fewer than 10 authors
            if not author_counts.empty:
                axs[1, 1].barh(author_counts.index, author_counts.values)
                axs[1, 1].set_title('Top Contributors')
                axs[1, 1].set_xlabel('Number of Commits')
        
        plt.tight_layout()
        return fig
    
    # Time series chart shows commits over time
    elif chart_type == "time_series":
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Group by date and count commits
        df_grouped = df.groupby(df['date'].dt.date).size().reset_index(name='count')
        df_grouped['date'] = pd.to_datetime(df_grouped['date'])
        df_grouped = df_grouped.sort_values('date')
        
        # Plot time series
        ax.plot(df_grouped['date'], df_grouped['count'], marker='o', linestyle='-', linewidth=2)
        ax.set_title('Commits Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Commits')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Format x-axis dates
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    # Author bar chart
    elif chart_type == "authors":
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Get top 15 authors
        author_counts = df['author'].value_counts().head(15)
        author_counts = author_counts.sort_values(ascending=True)
        
        # Plot horizontal bar chart
        ax.barh(author_counts.index, author_counts.values, color='skyblue')
        ax.set_title('Top Contributors')
        ax.set_xlabel('Number of Commits')
        
        # Add count numbers at the end of each bar
        for i, v in enumerate(author_counts.values):
            ax.text(v + 0.1, i, str(v), va='center')
            
        plt.tight_layout()
        return fig
    
    return None

def ask_gemini_about_commits(question: str, commits: list, output_mode: str):
    """Ask Gemini AI about the searched commits with specified output mode."""
    if not commits:
        return "No commits found to analyze. Please refine your search criteria."
    
    if 'GEMINI_API_KEY' not in st.secrets:
        return "Gemini API key not configured. Please add it to your Streamlit secrets."
    
    # Build comprehensive context from commits
    commit_summaries = []
    feature_areas = set()
    authors = set()
    time_periods = set()
    
    # Include a representative sample (most recent ones)
    sample_size = min(20, len(commits))
    commit_sample = commits[:sample_size]
    
    for commit in commit_sample:
        # Track metadata
        feature_areas.add(commit.get('feature_path', 'all'))
        authors.add(commit['author'])
        time_periods.add(commit['date'])
        
        # Enhanced commit summary
        summary = f"""
        Commit: {commit['sha'][:7]}
        Date: {commit['date']}
        Author: {commit['author']}
        Feature Area: {commit.get('feature_path', 'all')}
        Status: {"Feature/Enhancement" if commit.get('status', 'success') == "success" else "Bugfix/Error"}
        
        Message: {commit['message']}
        
        Changes: 
        - Files changed: {commit.get('files_changed', 0)}
        - Lines added: {commit.get('additions', 0)}
        - Lines removed: {commit.get('deletions', 0)}
        
        Modified Files:
        {', '.join(commit.get('file_paths', [])[:5])}{'...' if len(commit.get('file_paths', [])) > 5 else ''}
        """
        commit_summaries.append(summary)
    
    # Build metadata context
    metadata_context = f"""
    Analysis Context:
    - Feature Areas: {', '.join(feature_areas)}
    - Authors: {', '.join(authors)}
    - Time Period: {', '.join(sorted(time_periods))}
    - Total Commits Analyzed: {len(commits)} (showing details for {sample_size})
    - Success/Failure Ratio: {sum(1 for c in commits if c.get('status', 'success') == 'success')}/{sum(1 for c in commits if c.get('status', 'failure') == 'failure')}
    """
    
    # Build the prompt based on output mode
    mode_instructions = {
        "visualization": """
        Present your analysis as if you were creating visual representations. Describe:
        1. Key metrics that would be shown in charts
        2. Patterns that would be visible in the data
        3. Trends across time, authors, or feature areas
        4. Recommendations based on the visual patterns
        
        Structure your answer with "Visual Insight" sections.
        """,
        "errors_analysis": """
        Focus specifically on:
        1. Patterns in errors and fixes
        2. Most active feature areas
        3. Most productive authors
        4. Recommendations for quality improvements
        
        Structure with "Error Patterns", "Stable Components", and "Focus Areas".
        """,
        "tabular": """
        Present your answer as a structured table with:
        - Rows representing key insights
        - Columns for metrics, findings, and recommendations
        
        Format as a markdown table representation.
        """,
        "standard": """
        Provide a comprehensive analysis covering:
        1. Key changes and their impact
        2. Patterns in the commit history
        3. Notable contributors
        4. Recommendations for next steps
        """
    }.get(output_mode, "")
    
    # Build the final prompt
    prompt = f"""
    You are analyzing GitHub commits for a project manager. Here's the context:
    
    {metadata_context}
    
    Commit Details:
    {'---'.join(commit_summaries)}
    
    Question: {question}
    
    {mode_instructions}
    
    Requirements:
    - Focus on business impact, not technical details
    - Highlight patterns and trends
    - Provide actionable recommendations
    - Use simple language suitable for non-technical managers
    - Reference specific commits or authors when relevant
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Sorry, I couldn't generate a response: {str(e)}"

def cleanup_repo():
    """Clean up the cloned repository."""
    if st.session_state.repo_path and os.path.exists(st.session_state.repo_path):
        try:
            shutil.rmtree(st.session_state.repo_path, ignore_errors=True)
            st.session_state.repo_path = None
            st.session_state.current_repo_url = None
            st.session_state.branches = []
            st.session_state.tags = []
            st.session_state.total_commits = 0
        except Exception:
            pass

# ---- STREAMLIT UI ----

st.set_page_config(page_title="Commit Search for PMs", layout="wide")

# App title and description
st.title("🔍 Commit Explorer for Project Managers")
st.markdown("Search for commits in your GitHub repository without needing technical knowledge.")

# Sidebar for repo info and authentication
with st.sidebar:
    st.header("Repository Settings")
    repo_url = st.text_input("GitHub Repository URL", placeholder="https://github.com/user/repo")
    token = st.text_input("GitHub Personal Access Token (optional)", type="password", 
                           help="Needed for private repositories")
    
    connect_button_label = "Connect & Clone Repository"
    if st.session_state.current_repo_url == repo_url and st.session_state.repo_path:
        connect_button_label = "Refresh Repository"
        
    if repo_url and st.button(connect_button_label):
        try:
            # Clone or reuse the repository
            repo_path, branches, tags = clone_repository(repo_url, token)
            
            # Store in session state
            st.session_state.repo_path = repo_path
            st.session_state.branches = branches
            
            # Try to get tags from GitHub API first
            github_tags = get_github_tags(repo_url, token)
            if github_tags:
                st.session_state.tags = github_tags
            else:
                st.session_state.tags = tags
            
            # Get repository info
            repo_info = get_repo_info(repo_url, token)
            st.session_state.repo_info = repo_info
            
            action_verb = "Connected to" if st.session_state.current_repo_url == repo_url else "Cloned"
            st.success(f"{action_verb}: {repo_info['name']}")
            st.markdown(f"👥 **Owner**: {repo_info['owner']['login']}")
            st.markdown(f"⭐ **Stars**: {repo_info['stargazers_count']}")
            st.markdown(f"🍴 **Forks**: {repo_info['forks_count']}")
            st.markdown(f"📊 **Total Commits**: {st.session_state.total_commits}")
            
        except Exception as e:
            st.error(f"Failed to connect to repository: {e}")
    
    # Add clean up button if repo is cloned
    if st.session_state.repo_path:
        if st.button("Cleanup Repository"):
            cleanup_repo()
            st.success("Repository cleaned up!")
            st.experimental_rerun()

# Main search section
if st.session_state.repo_path:
    # Repository status indicator
    st.info(f"Currently using: {st.session_state.current_repo_url} ({st.session_state.total_commits} commits)")
    
    # Unified search form to link filters and search together
    with st.form("commit_search_form"):
        # Centered filters section
        st.subheader("Search Filters")
        
        # Get features from local repo
        features = get_features_from_local_repo(st.session_state.repo_path)
        
        # Three columns for filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_feature = st.selectbox("Feature Area", ["All Features"] + features)
            selected_branch = st.selectbox("Branch", ["All Branches"] + st.session_state.branches)
        
        with col2:
            selected_tag = st.selectbox("Tag", ["All Tags"] + st.session_state.tags)
            sha_code = st.text_input("Commit SHA (optional)", "")
        
        with col3:
            # Date range filter
            date_range = st.date_input("Date Range (optional)", [])
        
        st.divider()
        
        # Search input and button inside the form
        search_query = st.text_input("Search Commits", placeholder="Enter keywords (e.g., 'payment processing', 'login fix')")
        
        # Add a warning about large repositories
        if st.session_state.total_commits > 10000:
            st.warning(f"This repository has {st.session_state.total_commits} commits. Searches might take longer. Consider using more specific filters.")
        
        # Add option to limit results
        col1, col2 = st.columns(2)
        with col1:
            limit_results = st.checkbox("Limit results", value=True)
        with col2:
            if limit_results:
                max_results = st.slider("Maximum results", 100, 10000, 1000)
            else:
                max_results = 50000  # Hard limit to prevent memory issues
        
        # Submit button for the entire form
        submitted = st.form_submit_button("Search Commits", type="primary", use_container_width=True)
    
    # Process the search when form is submitted
    if submitted:
        # Determine which filters are active
        active_filters = []
        
        # Prepare filter parameters
        feature_path = None if selected_feature == "All Features" else selected_feature
        branch = None if selected_branch == "All Branches" else selected_branch
        tag = None if selected_tag == "All Tags" else selected_tag
        
        if feature_path:
            active_filters.append(f"Feature: {feature_path}")
        if branch:
            active_filters.append(f"Branch: {branch}")
        if tag:
            active_filters.append(f"Tag: {tag}")
        if sha_code:
            active_filters.append(f"Commit SHA: {sha_code[:8]}...")
        if date_range and len(date_range) == 2:
            active_filters.append(f"Date Range: {date_range[0]} to {date_range[1]}")
        if search_query:
            active_filters.append(f"Keywords: '{search_query}'")
        
        if not active_filters:
            st.warning("Please specify at least one filter or search term")
            st.stop()
        
        try:
            # Execute the search
            commits = search_local_commits(
                st.session_state.repo_path,
                search_query,
                feature_path=feature_path,
                branch=branch,
                tag=tag,
                sha=sha_code,
                date_range=date_range if date_range and len(date_range) == 2 else None
            )
            
            # Store ALL results in session state for LLM to use
            st.session_state.search_results = commits
            
            # Sort by date (newest first)
            commits.sort(key=lambda x: x["raw_date"], reverse=True)
            
            # Add to search history
            search_description = search_query if search_query else "Filtered search"
            st.session_state.search_history.append((search_description, datetime.now().strftime("%H:%M:%S")))
            
            if commits:
                st.success(f"Found {len(commits)} matching commits")
                
                # Show active filters
                with st.expander("Active Filters", expanded=False):
                    st.markdown("\n".join([f"- {f}" for f in active_filters]))
                
                # Add visualization tabs
                st.subheader("Commit Analysis")
                
                tab1, tab2, tab3 = st.tabs(["Summary", "Time Series", "Contributors"])
                
                with tab1:
                    commit_viz = create_visualization(commits, "summary")
                    if commit_viz:
                        st.pyplot(commit_viz)
                    else:
                        st.info("Not enough data for visualization")
                        
                with tab2:
                    time_viz = create_visualization(commits, "time_series")
                    if time_viz:
                        st.pyplot(time_viz)
                    else:
                        st.info("Not enough data for time series visualization")
                        
                with tab3:
                    author_viz = create_visualization(commits, "authors")
                    if author_viz:
                        st.pyplot(author_viz)
                    else:
                        st.info("Not enough data for author visualization")
                
                # Display commit list with pagination
                st.subheader("Commit List")
                
                # Pagination controls
                total_pages = (len(commits) + 9) // 10  # 10 commits per page
                
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    page = st.slider("Page", 1, max(1, total_pages), 1)
                
                # Calculate start and end indices
                start_idx = (page - 1) * 10
                end_idx = min(start_idx + 10, len(commits))
                
                st.text(f"Showing commits {start_idx+1}-{end_idx} of {len(commits)}")
                
                # Display the current page of commits
                for idx, commit in enumerate(commits[start_idx:end_idx]):
                    status_color = "🟢" if commit.get('status', 'success') == "success" else "🔴"
                    with st.expander(f"{status_color} [{commit['date']}] {commit['message'].splitlines()[0]}", expanded=(idx == 0)):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(f"**Author:** {commit['author']}")
                            st.markdown(f"**Date:** {commit['date']}")
                            st.markdown(f"**SHA:** {commit['sha'][:10]}...")
                            
                            # Generate summary
                            summary = generate_commit_summary(commit)
                            st.info(summary)
                                
                            st.markdown(f"**Full Message:**")
                            st.text(commit['message'])
                            
                        with col2:
                            st.markdown("**Stats:**")
                            st.markdown(f"- 📁 Files changed: {commit.get('files_changed', 0)}")
                            st.markdown(f"- ➕ Additions: {commit.get('additions', 0)}")
                            st.markdown(f"- ➖ Deletions: {commit.get('deletions', 0)}")
                            st.markdown(f"- Status: {'✅ Feature/Enhancement' if commit.get('status', 'success') == 'success' else '❌ Fix/Error'}")
                            
                            st.markdown("**Actions:**")
                            st.link_button("View on GitHub", commit['url'])
                
            else:
                st.warning(f"No commits found matching your criteria")
                
        except Exception as e:
            st.error(f"Error during search: {e}")

    # Add LLM Q&A section with output modes when search results are available
    if st.session_state.search_results:
        st.markdown("---")
        st.subheader("📋 Ask About These Commits")
        st.markdown("Ask questions about the commits you've just searched. The AI will analyze these filtered commits.")
        
        # Add output mode selection
        output_mode = st.radio("Select Output Mode", [
            "Standard (Text Response)", 
            "Visualization Analysis", 
            "Errors & Working Components Analysis", 
            "Tabular View"
        ], horizontal=True)
        
        # Map the radio options to internal mode values
        mode_mapping = {
            "Standard (Text Response)": "standard",
            "Visualization Analysis": "visualization",
            "Errors & Working Components Analysis": "errors_analysis",
            "Tabular View": "tabular"
        }
        selected_mode = mode_mapping[output_mode]
        
        llm_query = st.text_input("Your Question", placeholder="What was the main purpose of these changes?")
        if llm_query:
            with st.spinner("Analyzing commits and generating response..."):
                answer = ask_gemini_about_commits(llm_query, st.session_state.search_results, selected_mode)
                st.markdown("### Answer:")
                st.markdown(answer)
                
                # If using tabular view, also provide a clean data table
                if selected_mode == "tabular":
                    st.subheader("Data Table View")
                    # Create a DataFrame with commit info
                    df = pd.DataFrame([{
                        "Date": c["date"],
                        "Author": c["author"],
                        "Message": c["message"].splitlines()[0][:50] + "...",
                        "Files Changed": c.get("files_changed", 0),
                        "Additions": c.get("additions", 0),
                        "Deletions": c.get("deletions", 0),
                        "Type": "Feature/Enhancement" if c.get("status", "success") == "success" else "Fix/Error"
                    } for c in st.session_state.search_results[:1000]])  # Limit to 1000 rows for performance
                    
                    st.dataframe(df, use_container_width=True)
        
        # Small text to clarify what data is being used
        commit_count = len(st.session_state.search_results)
        st.caption(f"The AI is analyzing all {commit_count} commits from your filtered search results.")

# Search history in sidebar
with st.sidebar:
    if st.session_state.search_history:
        st.header("Recent Searches")
        for q, t in st.session_state.search_history[-5:]:  # Show last 5 searches
            st.markdown(f"• {t}: **{q}**")

# First-time instructions
if not st.session_state.repo_path:
    st.info("""
    ### How to Use This Tool:
    
    1. Enter your GitHub repository URL in the sidebar
    2. Add your GitHub Personal Access Token if needed for private repositories
    3. Click "Connect & Clone Repository" to create a local copy
    4. Use the filters to narrow your search by feature, branch, tag, SHA, or date range
    5. Type keywords in the search box and click "Search Commits"
    6. After finding commits, use the visualizations to gain high-level insights
    7. Use the AI Question box to ask specific questions about the filtered commits
    
    The tool will find relevant commits and provide visualizations and non-technical summaries.
    """)

# Register cleanup handler
import atexit
atexit.register(cleanup_repo)