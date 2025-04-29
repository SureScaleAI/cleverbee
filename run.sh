#!/bin/bash

# Ensure we are in the script's directory (project root)
cd "$(dirname "$0")"

# --- Argument Parsing ---
NO_CACHE_FLAG=false
CHAINLIT_ARGS="" # Store arguments for chainlit run

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --no-cache)
      NO_CACHE_FLAG=true
      shift # past argument
      ;;
    *) # unknown option, assume it's for chainlit run
      # Quote the argument to handle spaces correctly
      CHAINLIT_ARGS+="$1"" " 
      shift # past argument
      ;;
  esac
done
# --- End Argument Parsing ---

# Check if we're already in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
  # Activate virtual environment if it exists (optional, adjust path if needed)
  ACTIVATED_VENV=false
  # PRIORITIZE .venv first
  if [ -d ".venv" ]; then
    echo "Activating virtual environment (.venv)..."
    source .venv/bin/activate
    ACTIVATED_VENV=true
  elif [ -d "venv" ]; then
    echo "Activating virtual environment (venv)..."
    source venv/bin/activate
    ACTIVATED_VENV=true
  fi
else
  echo "Already in virtual environment: $VIRTUAL_ENV"
  ACTIVATED_VENV=false
fi

# Check if chainlit is installed within the environment
if ! command -v chainlit &> /dev/null
then
    echo "Warning: 'chainlit' command not found in the current environment."
    
    # Check if requirements.txt exists
    if [ -f "requirements.txt" ]; then
        echo "Attempting to install dependencies from requirements.txt..."
        
        # Check if pip is available
        if ! command -v pip &> /dev/null
        then
            echo "Error: 'pip' command not found. Cannot install dependencies."
            if [ "$ACTIVATED_VENV" = true ] && type deactivate &>/dev/null; then deactivate; fi
            exit 1
        fi

        # Run pip install
        pip install -r requirements.txt
        
        # Check the exit code of pip install
        if [ $? -ne 0 ]; then
            echo "Error: 'pip install -r requirements.txt' failed. Please check the output above and fix any issues manually."
            if [ "$ACTIVATED_VENV" = true ] && type deactivate &>/dev/null; then deactivate; fi
            exit 1
        else
            echo "Dependencies installed successfully."
            # Re-check if chainlit is available NOW
            if ! command -v chainlit &> /dev/null
            then
                 # Attempt to find chainlit executable path directly (might help in some cases)
                 CHAINLIT_PATH=$(which chainlit)
                 if [ -z "$CHAINLIT_PATH" ] || ! command -v chainlit &> /dev/null ; then
                     echo "Error: 'chainlit' command still not found after installing dependencies."
                     echo "Please check requirements.txt and ensure 'chainlit' is listed."
                     echo "Also check pip install output for errors."
                     if [ "$ACTIVATED_VENV" = true ] && type deactivate &>/dev/null; then deactivate; fi
                     exit 1
                 fi
                 # If which found it, try proceeding (though command -v should ideally work)
                 echo "'chainlit' command found after install (using which). Proceeding..."
            else
                 echo "'chainlit' command is now available. Proceeding to run the app..."
            fi
            # Chainlit is now available (or assumed available if `which` found it), script will continue
        fi
    else
        echo "Error: requirements.txt not found in the project root."
        echo "Cannot install dependencies. Please create requirements.txt or install manually."
        if [ "$ACTIVATED_VENV" = true ] && type deactivate &>/dev/null; then deactivate; fi
        exit 1
    fi
fi

# --- Cache Deletion Logic ---
DB_FILE=".langchain.db"
if [ "$NO_CACHE_FLAG" = true ]; then
    if [ -f "$DB_FILE" ]; then
        echo "Deleting cache file: $DB_FILE"
        rm -f "$DB_FILE"
        if [ $? -ne 0 ]; then
            echo "Warning: Failed to delete $DB_FILE. Proceeding anyway."
        fi
    else
        echo "Cache file $DB_FILE not found. Skipping deletion."
    fi
fi
# --- End Cache Deletion Logic ---

# --- Version (Tag) Based Update Detection ---
git fetch --tags
LOCAL_TAG=$(git describe --tags --abbrev=0 2>/dev/null)
REMOTE_TAG=$(git ls-remote --tags origin | awk -F/ '{print $3}' | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | sort -V | tail -n1)

if [ -z "$REMOTE_TAG" ]; then
  echo "No remote version tags found. Skipping version update check."
  REPO_UPDATED=false
elif [ "$LOCAL_TAG" != "$REMOTE_TAG" ]; then
  echo "A new version ($REMOTE_TAG) is available. You are on $LOCAL_TAG."
  read -p "Do you want to update to the latest version? (Y/n): " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ || -z $REPLY ]]; then
    # Only check for uncommitted changes if user wants to update
    if ! git diff-index --quiet HEAD --; then
      echo "You have uncommitted changes. Please commit or stash them before updating the repo."
      exit 1
    fi
    git checkout "$REMOTE_TAG" || { echo "Failed to checkout $REMOTE_TAG. Please resolve manually."; exit 1; }
    REPO_UPDATED=true
  else
    echo "Skipping update. Running with current version ($LOCAL_TAG)."
    REPO_UPDATED=false
  fi
else
  REPO_UPDATED=false
fi

# --- Check if requirements.txt changed after pull ---
REQUIREMENTS="requirements.txt"
HASH_FILE=".venv/.requirements_hash"
if [ -f "$REQUIREMENTS" ]; then
  CURRENT_HASH=$(shasum -a 256 "$REQUIREMENTS" | awk '{print $1}')
else
  CURRENT_HASH=""
fi
NEEDS_INSTALL=false
if [ ! -f "$HASH_FILE" ]; then
  NEEDS_INSTALL=true
elif [ -n "$CURRENT_HASH" ]; then
  STORED_HASH=$(cat "$HASH_FILE")
  if [ "$CURRENT_HASH" != "$STORED_HASH" ]; then
    NEEDS_INSTALL=true
  fi
fi

if [ "$REPO_UPDATED" = true ] && [ "$NEEDS_INSTALL" = true ]; then
  read -p "requirements.txt has changed. Update dependencies? (Y/n): " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ || -z $REPLY ]]; then
    pip install --upgrade -r "$REQUIREMENTS"
    if [ $? -eq 0 ]; then
      echo "$CURRENT_HASH" > "$HASH_FILE"
      echo "Dependencies updated."
    else
      echo "Dependency installation failed. Please check the output above."
      exit 1
    fi
  else
    echo "Skipping dependency update. The application may not run correctly."
  fi
fi

# Detect the correct python executable (prefer python3, fallback to python)
if command -v python3 &> /dev/null; then
  PYTHON_CMD=python3
elif command -v python &> /dev/null; then
  PYTHON_CMD=python
else
  echo "Error: Neither 'python3' nor 'python' was found in PATH. Please install Python 3.8+ and try again."
  if [ "$ACTIVATED_VENV" = true ] && type deactivate &>/dev/null; then deactivate; fi
  exit 1
fi

# If chainlit command was found (either initially or after install), run the application
# -w flag enables auto-reloading, remove if not desired for production
# Use eval to correctly handle quoted arguments in CHAINLIT_ARGS
echo "Starting Chainlit app (src/chainlit_app.py) with args: $CHAINLIT_ARGS..."
eval $PYTHON_CMD -m chainlit run src/chainlit_app.py $CHAINLIT_ARGS

# Deactivate virtual environment (if applicable)
# Check if deactivate function exists before calling
if [ "$ACTIVATED_VENV" = true ] && type deactivate &>/dev/null; then
  echo "Deactivating virtual environment."
  deactivate
fi

echo "Chainlit app stopped." 