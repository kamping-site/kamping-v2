# Sphinx configuration for the KaMPIng-v2 documentation.
#
# Build with `cmake --build <build-dir> --target docs` (configure with -DKAMPING_V2_BUILD_DOCS=ON). CMake runs Doxygen
# and passes the location of its XML output in KAMPING_V2_DOXYGEN_XML; _ext/kamping_doxygen.py renders the API from it.

import os
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent
DOXYGEN_XML = os.environ.get("KAMPING_V2_DOXYGEN_XML")
if not DOXYGEN_XML:
    sys.exit("KAMPING_V2_DOXYGEN_XML is not set: build the docs with `cmake --build <build-dir> --target docs`")

sys.path.insert(0, str(DOCS / "_ext"))

project = "KaMPIng-v2"
copyright = "2026, Karlsruhe Institute of Technology"
author = "The KaMPIng authors"

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "kamping_doxygen",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
exclude_patterns = ["_build", "CMakeLists.txt"]

# --- MyST ---
myst_enable_extensions = ["colon_fence", "deflist", "fieldlist"]
myst_heading_anchors = 3

# --- C++ ---
primary_domain = "cpp"
highlight_language = "cpp"
toc_object_entries_show_parents = "hide"  # "resize_t", not "kamping::v2::resize_t", in the page outline

# --- API from Doxygen (_ext/kamping_doxygen.py) ---
kamping_doxygen_xml = DOXYGEN_XML
kamping_doxygen_v2_include = str(DOCS.parent / "kamping-v2" / "include")  # its `using` declarations

nitpicky = True  # references written in the pages must resolve


# --- HTML ---
html_theme = "furo"
html_title = "KaMPIng-v2"
html_logo = "icon.svg"
html_static_path = ["_static"]
html_css_files = ["kit.css"]

# KIT colours: green for the brand and links, lilac for visited links.
_kit_green, _kit_lilac = "#009682", "#A3107C"
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": _kit_green,
        "color-brand-content": _kit_green,
        "color-brand-visited": _kit_lilac,
        "color-api-name": _kit_green,
        "color-api-pre-name": _kit_green,
        "color-highlight-on-target": "rgba(0, 150, 130, 0.12)",
    },
    "dark_css_variables": {
        "color-brand-primary": _kit_green,
        "color-brand-content": _kit_green,
        "color-brand-visited": "#D46FB8",  # KIT lilac lightened for dark backgrounds
        "color-api-name": _kit_green,
        "color-api-pre-name": _kit_green,
        "color-highlight-on-target": "rgba(0, 150, 130, 0.25)",
    },
    "source_repository": "https://github.com/kamping-site/kamping-v2",
    "source_branch": "main",
    "source_directory": "docs/",
}
