# Project Overview

This project implements a RAG system that uses ArangoDB as the knowledge graph backend. It allows querying data using AQL (ArangoDB Query Language) through a Python interface and integrates with large language models like OpenAI and Anthropic for advanced query processing.

## GAIA: Graph Analysis Intelligence Agent

GAIA (Graph Analytics Intelligence Agent) is an AI-powered analytics platform that transforms natural language questions into complex graph queries and visualizations. Using a Steam gaming dataset with users, games, and playtime relationships, GAIA can:

- Perform complex network analysis like community detection, centrality calculations, and path finding
- Generate interactive visualizations showing relationships between games and players
- Identify game recommendations based on network patterns
- Detect communities of players with similar tastes
- Find "bridge games" that connect different gaming communities
- Analyze user playing patterns and statistics
- Visualize the structure of gaming ecosystems through force-directed graphs, heatmaps, and other specialized visualizations

Users simply ask questions in plain English, and GAIA handles the complex graph querying, algorithm selection, and visualization creation automatically.

## Environment Setup

### Prerequisites

- Python 3.12+
- Conda (recommended for managing the environment)
- ArangoDB account (cloud or self-hosted)
- OpenAI API key
- Anthropic API key (optional)

### Setting up the environment

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/ArangoGraphRag.git
   cd ArangoGraphRag
   ```

2. Create and activate the conda environment:

   ```bash
   conda env create -f environment.yml
   conda activate base  # The environment is named 'base' in the yml file
   ```

3. Install additional dependencies (if not already included in environment.yml):
   ```bash
   pip install langchain langchain_community python-dotenv arango openai anthropic
   ```

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```
# ArangoDB Configuration
ARANGO_HOST="your_arango_host_url"
ARANGO_USER="your_arango_username"
ARANGO_PASSWORD="your_arango_password"
ARANGO_DB="your_arango_database_name"

# OpenAI Configuration
OPENAI_API_KEY="your_openai_api_key"

# Anthropic Configuration (optional)
ANTHROPIC_API_KEY="your_anthropic_api_key"
```

Replace the placeholder values with your actual credentials.

## Project Structure

- **ArangoGraphDirectChain.py**: A custom LangChain tool that allows querying ArangoDB using AQL with dynamic variables binding.
- **agent_utils.py**: The GAIA agent implementation.
- **utils.py**: Miscellaneous utility functions
- **program.ipynb**: Jupyter notebook for interactive usage
- **DataPrep.ipynb**: Notebook for data preparation and persistence into ArangoDB
