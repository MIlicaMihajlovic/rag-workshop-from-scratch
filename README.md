# rag-demo

A bare bones RAG application for educational purposes.

DISCLAIMER: There are several concepts in this repository that can be implemented in much better ways.  The point of this repository is to remove unfamiliar terms and abstractions as much as possible to demonstrate the essential concepts of a RAG application.

You should get acquainted first with RAG and [when to use it and when not to.](https://www.anthropic.com/news/contextual-retrieval)
Also, feel free to check out the [BGE family of models](https://huggingface.co/BAAI/bge-small-en-v1.5) a series of API accessible models for many RAG pieces such as embeddings, retrieval, reranking, etc. 

## Prerequisites

- **Python 3.12 or higher**: Ensure you have Python 3.12 or a later version installed.
- **Poetry**: Install Poetry on your machine for dependency management.
- **Docker Desktop for Mac**: Install Docker Desktop and ensure Docker Engine is running.
- **SSH Key** (optional): Configure your SSH key with your GitHub account if using SSH clone.

Download Links:
- [Download Docker Desktop for Mac](https://www.docker.com/products/docker-desktop/)
- [Download Python 3.12](https://www.python.org/downloads/)
- [Install Poetry (Official Installer)](https://python-poetry.org/docs/#installing-with-the-official-installer)
- [Generate SSH Key (if needed)](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)


## Sample output
```
Cleaning database...
Creating embedding for chunk: ize
human priors to ...
Creating embedding for chunk: odifications, ensuri...
Creating embedding for chunk: u et al., 2023; Qian...
Creating embedding for chunk: ned agents and meta-...
Creating embedding for chunk: n Hendrycks, Collin ...

Total index time: 11.419859999790788ms

Enter question: what are the main contributions from the godel agents paper?
scores:  [46.2205144035501, 45.83206449938195, 44.91440161297333, 44.8048227792767, 44.57226661278396]

Using 5 chunks in answer. Answer:

Compiling declarative language model calls into self-improving pipelines
```

## Setup (macOS, No Homebrew Required):

### Step 1: Install Docker Desktop
1. Go to: https://www.docker.com/products/docker-desktop/
2. Download Docker Desktop for Mac (Apple Silicon or Intel).
3. Open the .dmg file and drag Docker to Applications.
4. Launch Docker from Applications.
5. Wait until Docker is running (check menu bar).

Verify:
```bash
docker --version
docker run --rm hello-world
```

### Step 2: Install Python 3.12
1. Go to: https://www.python.org/downloads/
2. Download Python 3.12 macOS installer (.pkg).
3. Run the installer and complete the setup.

Verify:
```bash
python3 --version
```

### Step 3: Install Poetry
```bash
curl -sSL https://install.python-poetry.org | python3 -
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
poetry --version
```

### Step 4: Clone the Repository

Using SSH (if configured):
```bash
git clone git@github.com:smferro54/rag-workshop-from-scratch.git
cd rag-workshop-from-scratch
```

Or using HTTPS:
```bash
git clone https://github.com/smferro54/rag-workshop-from-scratch.git
cd rag-workshop-from-scratch
```

### Step 5: Start pgvector Database Container
```bash
docker run -p 6432:5432 --name pgvector -e POSTGRES_PASSWORD=postgres -d pgvector/pgvector:pg17
```

Verify container is running:
```bash
docker ps
docker logs pgvector --tail 50
```

### Step 6: Create Database and Apply Schema

No need to install local psql—use Docker to execute commands:

Create database:
```bash
docker exec -it pgvector psql -U postgres -c "CREATE DATABASE rag_demo;"
```

Apply schema:
```bash
docker exec -i pgvector psql -U postgres -d rag_demo < schema.sql
```

Verify:
```bash
docker exec -it pgvector psql -U postgres -d rag_demo -c "SELECT * FROM chunks LIMIT 5;"
```

### Step 7: Create .env File
```bash
cp example_env.txt .env
```

Edit `.env` and add your Hugging Face API key:
```bash
source .env
```

### Step 8: Install Python Dependencies
```bash
poetry install
```

### Step 9: Run the Application
```bash
poetry run python -m rag_demo
```

Or skip embedding if database already indexed:
```bash
poetry run python -m rag_demo --skip-embedding-step
```

### Useful Docker Commands

Stop container:
```bash
docker stop pgvector
```

Start container:
```bash
docker start pgvector
```

Remove container (fresh restart):
```bash
docker rm -f pgvector
```

Then recreate with Step 5 command.

## Troubleshooting

**Docker won't start:**
- Restart Docker Desktop from Applications.
- Check System Preferences > Security & Privacy (grant permissions if prompted).
- Reboot your Mac if services are stuck.

**Python version issues:**
- Verify `python3 --version` shows 3.12+.
- If not, use the full path: `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 --version`
- Or reinstall Python 3.12 from python.org.

**Port 6432 already in use:**
- Use a different port: `docker run -p 7432:5432 --name pgvector -e POSTGRES_PASSWORD=postgres -d pgvector/pgvector:pg17`
- Update connection commands to use 7432 instead of 6432.

**Poetry not found:**
- Verify Poetry installed: `poetry --version`
- Reload shell: `source ~/.zshrc`

**Docker exec psql commands fail:**
- Verify container is running: `docker ps`
- Check container logs: `docker logs pgvector`
- Recreate container: `docker rm -f pgvector` then re-run Step 5.
