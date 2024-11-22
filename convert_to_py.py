import nbformat
from nbconvert import PythonExporter

def convert_ipynb_to_py(ipynb_file, py_file):
    # Read the notebook file
    with open(ipynb_file, 'r', encoding='utf-8') as f:
        notebook_content = nbformat.read(f, as_version=4)

    # Convert the notebook content to a Python script
    python_exporter = PythonExporter()
    script, _ = python_exporter.from_notebook_node(notebook_content)

    # Write the Python script to a file
    with open(py_file, 'w', encoding='utf-8') as f:
        f.write(script)

# Example usage 
ipynb_file = './playground-series-s4e11/94-344-lgbm-bayessearchcv-sin-transformer.ipynb'  # Replace with your .ipynb file path
py_file = './playground-series-s4e11/94-344-lgbm-bayessearchcv-sin-transformer.py'          # Replace with your desired .py file path
convert_ipynb_to_py(ipynb_file, py_file)
