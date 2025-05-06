import json
from tqdm import tqdm
from tree_sitter import Parser, Language
import tree_sitter_python
import ast
from repo_setup import AirflowRepoManager
from datasets import load_dataset

MIN_DOCSTRING_TOKENS = 20
MAX_CLASS_LINES = 400

# TreeSplitter Configurations
PYTHON_LANGUAGE = Language(tree_sitter_python.language())
parser = Parser(PYTHON_LANGUAGE)

root_dir = AirflowRepoManager().clone_repo()

records = []
total_doc_tokens = 0
total_code_tokens = 0
num_samples = 0

for py_file in tqdm(root_dir.rglob("*.py"), desc="Parsing Python files"):
    # Skip tests directories
    if any(part in {"tests"} for part in py_file.parts):
        continue

    with py_file.open("rb") as f:
        code_bytes = f.read()

    try:
        tree = parser.parse(code_bytes)
    except Exception as e:
        print(f"Failed to parse {py_file}: {e}")
        continue
    
    def find_classes(node):
        if node.type == "class_definition":
            yield node
        for child in node.children:
            yield from find_classes(child)

    for class_node in find_classes(tree.root_node):
        name_node = class_node.child_by_field_name("name")
        class_name = name_node.text.decode("utf-8") if name_node else None

        parent_node = class_node.child_by_field_name("superclasses")
        if parent_node:
            parents = [c.text.decode("utf-8") for c in parent_node.children if c.type != ","]
        else:
            parents = []

        block_node = class_node.child_by_field_name("body")
        if not block_node:
            continue

        # Get first statement to find docstring
        meaningful_children = [c for c in block_node.children if c.type not in {"comment", "newline"}]
        docstring_text = None
        string_node = None 

        if meaningful_children:
            first_stmt = meaningful_children[0]
            if first_stmt.type == "expression_statement":
                string_node = next(
                    (c for c in first_stmt.children if c.type == "string"),
                    None
                )
                if string_node:
                    doc_bytes = code_bytes[string_node.start_byte: string_node.end_byte]
                    try:
                        docstring_text = ast.literal_eval(doc_bytes.decode("utf-8"))
                    except Exception:
                        continue  # Skip malformed strings

        # Skip if docstring is missing or trivial
        if not docstring_text or len(docstring_text.strip().split()) < MIN_DOCSTRING_TOKENS:
            continue

        # Extract the full class source code (with decorators)
        class_start = class_node.start_byte
        class_end = class_node.end_byte

        # Remove the entire expression_statement that wraps the docstring (if found)
        if meaningful_children:
            first_stmt = meaningful_children[0]
            if first_stmt.type == "expression_statement" and string_node:
                expr_start = first_stmt.start_byte
                expr_end = first_stmt.end_byte

                before_doc = code_bytes[class_start: expr_start]
                after_doc = code_bytes[expr_end: class_end]

                cleaned_class_bytes = before_doc + after_doc
            else:
                cleaned_class_bytes = code_bytes[class_start:class_end]
        else:
            cleaned_class_bytes = code_bytes[class_start:class_end]

        body_text = cleaned_class_bytes.decode("utf-8")

        # Truncate long classes (optional)
        body_lines = body_text.splitlines()
        if len(body_lines) > MAX_CLASS_LINES:
            body_text = "\n".join(body_lines[:MAX_CLASS_LINES]) + "\n# ...(truncated)..."

        record = {
            "module_path": str(py_file.relative_to("airflow-repository")),
            "class_name": f"{py_file.stem}.{class_name}",
            "parent_class": parents,
            "source_code": body_text.strip(),
            "docstring": docstring_text.strip(),
        }
        records.append(record)

        doc_tokens = len(record["docstring"].split())
        code_tokens = len(record["source_code"].split())

        total_doc_tokens += doc_tokens
        total_code_tokens += code_tokens
        num_samples += 1


if num_samples > 0:
    avg_doc_tokens = total_doc_tokens / num_samples
    avg_code_tokens = total_code_tokens / num_samples

    print(f"Average docstring tokens: {avg_doc_tokens:.2f}")
    print(f"Average code tokens: {avg_code_tokens:.2f}")

    with open("dataset_summary.txt", "w", encoding="utf-8") as summary_file:
        summary_file.write(f"Total samples: {num_samples}\n")
        summary_file.write(f"Average docstring tokens: {avg_doc_tokens:.2f}\n")
        summary_file.write(f"Average code tokens: {avg_code_tokens:.2f}\n")
else:
    print("No samples collected.")

with open("dataset.json", "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print("✅ Dataset creation complete.")

# Loading dataset to HuggingFace
dataset = load_dataset("json", data_files={
    "train": "dataset.json"
})

dataset.push_to_hub("anaterna/airflow-class-summarization")

print("Dataset pushed to HuggingFace.")