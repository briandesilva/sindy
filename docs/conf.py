import importlib
import os
import re
import shutil
from pathlib import Path
from typing import TypeVar

import requests
import yaml
from docutils import nodes
from docutils.statemachine import StringList
from sphinx.application import Sphinx
from sphinx.directives.other import TocTree
from sphinx.util.docutils import SphinxDirective

author = "dynamicslab"
project = "pysindy"  # package name


# no need to edit below this line

copyright = f"2020, {author}"

module = importlib.import_module(project)
version = release = getattr(module, "__version__")

master_doc = "index"

extensions = [
    "nbsphinx",
    "sphinxcontrib.apidoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "IPython.sphinxext.ipython_console_highlighting",
]

nb_execution_mode = "off"

apidoc_module_dir = f"../{project}"
apidoc_excluded_paths = ["tests"]
apidoc_toc_file = False

autodoc_default_options = {"members": True}
autodoc_member_order = "bysource"
autoclass_content = "init"

language = "en"

here = Path(__file__).parent.resolve()

exclude_patterns = ["build", "_build", "Youtube"]
# pygments_style = "sphinx"

add_module_names = True
add_function_parentheses = False
todo_include_todos = True

html_theme = "sphinx_rtd_theme"
html_show_sourcelink = False
html_show_sphinx = False
html_show_copyright = True

default_role = "any"
html_sourcelink_suffix = ""

intersphinx_mapping = {
    "derivative": ("https://derivative.readthedocs.io/en/latest/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# -- Extensions to the  Napoleon GoogleDocstring class ---------------------
# michaelgoerz.net/notes/extending-sphinx-napoleon-docstring-sections.html
from sphinx.ext.napoleon.docstring import GoogleDocstring  # noqa: E402


def parse_keys_section(self, section):
    return self._format_fields("Keys", self._consume_fields())


GoogleDocstring._parse_keys_section = parse_keys_section


def parse_attributes_section(self, section):
    return self._format_fields("Attributes", self._consume_fields())


GoogleDocstring._parse_attributes_section = parse_attributes_section


def parse_class_attributes_section(self, section):
    return self._format_fields("Class Attributes", self._consume_fields())


GoogleDocstring._parse_class_attributes_section = parse_class_attributes_section


def patched_parse(self):
    """
    we now patch the parse method to guarantee that the the above methods are
    assigned to the _section dict
    """
    self._sections["keys"] = self._parse_keys_section
    self._sections["class attributes"] = self._parse_class_attributes_section
    self._unpatched_parse()


GoogleDocstring._unpatched_parse = GoogleDocstring._parse
GoogleDocstring._parse = patched_parse


def setup(app: Sphinx):
    """Our sphinx extension for copying from examples/ to docs/examples

    Since nbsphinx does not handle glob/regex paths, we need to
    manually copy documentation source files from examples.  See issue
    # 230.
    """
    doc_examples = here / "examples"
    if not doc_examples.exists():
        (here / "examples").mkdir()
    example_source = (here / "../examples").resolve()
    source_notebooks = example_source.glob("**/*.ipynb")
    shutil.copy(example_source / "README.rst", doc_examples / "index.rst")
    for notebook in source_notebooks:
        if notebook.parent == example_source:
            new_dir = doc_examples / notebook.stem
        else:
            new_dir = doc_examples / notebook.parent.stem
        new_dir.mkdir(exist_ok=True)
        new_file = new_dir / "example.ipynb"
        print(f"Copying {notebook} to {new_file}")
        shutil.copy(notebook, new_file)
    # Notebook 15 uses an image file
    (doc_examples / "15_pysindy_lectures/data").mkdir(exist_ok=True)
    shutil.copy(
        example_source / "data/optimizer_summary.jpg",
        doc_examples / "15_pysindy_lectures/data/optimizer_summary.jpg",
    )
    if (here / "static/custom.css").exists():
        app.add_css_file("custom.css")

    _grab_external_examples(example_source)
    app.add_directive("pysindy-example", PysindyExample)


EXTERNAL_EXAMPLES: dict[str, list[tuple[str, str]]] = {}


def _load_ext_config(example_source: Path) -> list[dict[str, str]]:
    ext_config = example_source / "external.yml"
    with open(ext_config) as f:
        ext_examples = yaml.safe_load(f)
    return ext_examples


def _grab_external_examples(example_source: Path):
    ext_examples = _load_ext_config(example_source)
    for example in ext_examples:
        ex_name = example["name"]
        user = example["user"]
        repo = example["repo"]
        ref = example["ref"]
        dir = example["dir"]
        base = f"https://raw.githubusercontent.com/{user}/{repo}/{ref}/{dir}/"
        notebooks = fetch_notebook_list(base)
        base = f"https://raw.githubusercontent.com/{user}/{repo}/{ref}/"
        local_nbs = [(name, copy_nb(base, pth, repo)) for name, pth in notebooks]
        EXTERNAL_EXAMPLES[ex_name] = local_nbs


class PysindyExample(SphinxDirective):
    required_arguments = 0
    optional_arguments = 0
    option_spec = {"key": str, "title": str}
    has_content = True

    def run(self) -> list[nodes.Node]:
        key = self.options["key"]
        example_config = _load_ext_config((here / "../examples").resolve())
        try:
            this_example = [ex for ex in example_config if ex["name"] == key][0]
        except IndexError:
            RuntimeError("Unknown configuration key for external example")
        heading_text: str = self.options.get("title")
        base_repo = f"https://github.com/{this_example['user']}/{this_example['repo']}"
        repo_ref = nodes.reference(name="Source repo", refuri=base_repo)
        ref_text = nodes.Text("Source repo")
        repo_ref += ref_text
        repo_par = nodes.paragraph()
        repo_par += repo_ref
        normalized_text = re.sub(r"\s", "_", heading_text)
        tgt_node = nodes.target(refid=normalized_text)
        title_node = nodes.title()
        title_text = nodes.Text(heading_text)
        title_node += [title_text, tgt_node]
        content_nodes = self.parse_content_to_nodes()
        toc_items = []
        for name, relpath in EXTERNAL_EXAMPLES[key]:
            if name:
                toc_str = f"{name} <{relpath}>"
            if not name:
                toc_str = relpath
            toc_items.append(toc_str)
        toc_nodes = TocTree(
            name="PysindyExample",
            options={"maxdepth": 1},
            arguments=[],
            content=StringList(initlist=toc_items),
            lineno=self.lineno,
            block_text="",
            content_offset=self.content_offset,
            state=self.state,
            state_machine=self.state_machine,
        ).run()
        section_node = nodes.section(ids=[heading_text], names=[heading_text])
        section_node += [title_node, *content_nodes, *toc_nodes, repo_par]
        return [section_node]


def fetch_notebook_list(base: str) -> list[tuple[str, str]]:
    """Gets the list of example notebooks from a repo's index.html

    Each entry is a tuple of the title name of a link and the address
    """
    index = requests.get(base + "index.rst")
    if index.status_code != 200:
        raise RuntimeError("Unable to locate external example directory")
    text = str(index.content, encoding="utf-8")
    link_line = r"^\s+(.*)[^\S\r\n]+(\S+.ipynb)"
    T = TypeVar("T")

    def deduplicate(mylist: list[T]) -> list[T]:
        return list(set(mylist))

    rellinks = deduplicate(re.findall(link_line, text, flags=re.MULTILINE))
    return rellinks


def copy_nb(base: str, relpath: str, repo: str) -> str:
    """Create a local copy of external file, modifying relative reference"""
    example_dir = Path(__file__).parent / "examples"
    repo_local_dir = example_dir / repo
    repo_local_dir.mkdir(exist_ok=True)
    page = requests.get(base + relpath)
    if page.status_code != 200:
        raise RuntimeError(f"Unable to locate external notebook at {base + relpath}")
    filename = repo_local_dir / relpath.rsplit("/", 1)[1]
    with open(filename, "wb") as f:
        f.write(page.content)
    return os.path.relpath(filename, start=example_dir)
