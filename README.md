# NGHackWeekTeam4

### Pip

#### Windows

    ```bash
    python -m venv nghack
    nghack\Scripts\activate
    ```

#### Mac

    ```bash
    python3 -m venv nghack
    source nghack/bin/activate
    ```

### Conda

#### Create a Conda Environment

**Windows & Mac**:

    ```bash
    conda create --name nghack python=3.x

    ```

Replace `3.x` with your desired Python version.

#### Activate the Environment

    ```bash
    conda activate nghack  
    ```

Replace `myenv` with any name you prefer for your virtual environment.

### 📦 Installing Packages

Once you've activated your virtual environment, you should install the necessary Python packages provided in the requirements.txt file in the assignment folder. This ensures that you have all the necessary dependencies for the assignment.

**For pip users:**

    ```
    pip install -r requirements.txt
    ```

**For Conda users:**

    ```
    pip install -r requirements.txt
    ```
this is because `conda` does not directly support the `-r` flag with requirements.txt files.
