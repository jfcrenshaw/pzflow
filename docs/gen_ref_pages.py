"""Generate the code reference pages."""

from pathlib import Path

import mkdocs_gen_files

# recurse through python files in pzflow
for path in sorted(Path("pzflow").rglob("*.py")):
    # get the module path, not including the pzflow prefix
    module_path = path.with_suffix("")
    # path where we'll save the markdown file for this module
    doc_path = "API" / path.relative_to("pzflow").with_suffix(".md")

    # split up the path into its parts
    parts = list(module_path.parts)

    # we don't want to explicitly list __init__
    if parts[-1] in ["__init__"]:
        #continue
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
    # skip main files
    elif parts[-1] == "__main__":
        continue

    # create the markdown file in the docs directory
    with mkdocs_gen_files.open(doc_path, "w") as fd:
        identifier = ".".join(parts)
        print(doc_path, identifier)
        print("::: " + identifier, file=fd)

    mkdocs_gen_files.set_edit_path(doc_path, path)
