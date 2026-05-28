# Upload Another HPC Folder to GitHub

This workflow uploads a project folder from the HPC to GitHub while avoiding
large generated files that GitHub usually should not store, such as virtual
environments, caches, temporary folders, logs, and model checkpoints.

## 1. Choose the Folder and Repository

Set these values first:

```bash
PROJECT_DIR=/path/to/your/hpc/folder
REPO_URL=https://github.com/pppppppppppp-North/drl.git
```

For a different GitHub repository, replace `REPO_URL` with that repository's
HTTPS URL.

## 2. Go to the Project Folder

```bash
cd "$PROJECT_DIR"
```

## 3. Add a `.gitignore`

Create or update `.gitignore` so generated files are skipped:

```bash
cat > .gitignore <<'EOF'
.agents/
.codex/
.conda_envs/
.conda_pkgs/
.npm-cache/
.npm-global/
local/
tmp/

venv/
.venv/
env/
__pycache__/
*.py[cod]

*.out
*.err
._*

checkpoints/
data/cached/
*.zip
*.tar.gz
EOF
```

Edit this file if the project has important data that should be uploaded.
Do not upload files larger than 100 MB to normal GitHub unless you set up
Git LFS.

## 4. Initialize Git

Most folders can use normal Git:

```bash
git init
git branch -M main
```

If the folder has a broken or read-only `.git` directory, use a temporary Git
metadata directory instead:

```bash
GIT_DIR_TMP=/tmp/upload-$(basename "$PROJECT_DIR").git
git --git-dir="$GIT_DIR_TMP" --work-tree="$PROJECT_DIR" init
git --git-dir="$GIT_DIR_TMP" --work-tree="$PROJECT_DIR" branch -M main
```

For the remaining commands, use either normal `git` or the temporary form.

Normal form:

```bash
git status --short --ignored
git add .
git commit -m "Initial project upload"
git remote add origin "$REPO_URL"
git push -u origin main
```

Temporary Git directory form:

```bash
git --git-dir="$GIT_DIR_TMP" --work-tree="$PROJECT_DIR" status --short --ignored
git --git-dir="$GIT_DIR_TMP" --work-tree="$PROJECT_DIR" add .
git --git-dir="$GIT_DIR_TMP" --work-tree="$PROJECT_DIR" commit -m "Initial project upload"
git --git-dir="$GIT_DIR_TMP" --work-tree="$PROJECT_DIR" remote add origin "$REPO_URL"
git --git-dir="$GIT_DIR_TMP" --work-tree="$PROJECT_DIR" push -u origin main
```

## 5. Authenticate

GitHub no longer accepts account passwords for HTTPS pushes. Use a personal
access token when prompted for the password.

Create a token here:

```text
https://github.com/settings/tokens
```

For a fine-grained token, allow access to the target repository and give it:

```text
Contents: Read and write
```

After using a token in a shared terminal or chat, revoke it and create a new one
for the next upload.

## 6. Verify the Upload

Open the repository in a browser:

```text
https://github.com/pppppppppppp-North/drl
```

Check that source files are present and generated folders such as `venv`,
`tmp`, caches, logs, and checkpoints were not uploaded.

