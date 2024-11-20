![Python](https://img.shields.io/badge/python-3.13-blue.svg)
![License](https://img.shields.io/github/license/ovgarol/detritus-logistic-classification)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)

# detritus-logistic-classification
Logistic model for detritus classification in FlowCAM data.
This simple method gives good results for a limited dataset used [here](https://doi.org/10.1093/plankt/fbac013).
A more comprehensive description of the method and its results is [here](https://doi.org/10.1101/2024.11.18.624123).

# Installation and setup
To run the jupyter notebook, follow these steps to set up your environment:

1. Clone the repository:
    ```bash
    git clone https://github.com/ovgarol/detritus-logistic-classification.git
    ```

2. Navigate to the project directory:
    ```bash
    cd detritus-logistic-classification
    ```

3. Create a virtual environment (optional but recommended):
    ```bash
    python3 -m venv env
    ```

4. Activate the virtual environment:
    - For Windows:
      ```bash
      .\env\Scripts\activate
      ```
    - For macOS and Linux:
      ```bash
      source env/bin/activate
      ```

5. Install the required dependencies:
    ```bash
    pip install --upgrade pip
    pip install uv
    uv pip install -r requirements.txt
    ```

Now you should be able to run the jupyter notebook

