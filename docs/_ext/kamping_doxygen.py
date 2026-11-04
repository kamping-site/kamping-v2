"""Sphinx directives that render the KaMPIng-v2 API from the Doxygen XML.

Doxygen only parses the headers; these directives turn its XML into C++ domain entries (indexed, cross-referenced),
with the Doxygen comment as their description:

``doxygenclass`` / ``doxygenstruct``
    A class with its comment; with ``:members:`` also its documented public members, including inherited ones.

``doxygenoverloads``
    All overloads of a function. As on cppreference, the signatures are listed in one block and numbered; below them,
    each description is labelled with the overloads it applies to, and the parameters and return values are shared.

``doxygenconcept``
    A concept with its definition as written in the header.

``doxygenvariable`` / ``doxygentypedef``
    A variable or constant, without ``inline constexpr``; a type alias.

``doxygenview``
    The ``inline constexpr`` adaptor objects in ``kamping::v2::views``. Their declared type is an adaptor template
    over a lambda, which is useless to a reader, so the signatures shown are taken from a ``@code{.signature}`` block
    in the comment, one declaration per line (indented lines continue the previous one)::

        /// Attaches per-rank counts.
        ///
        /// @code{.signature}
        /// template <std::ranges::range Counts> auto with_counts(Counts&& counts);
        /// template <typename Buf, std::ranges::range Counts> auto with_counts(Buf&& buf, Counts&& counts);
        /// @endcode
        inline constexpr kamping::v2::adaptor<...> with_counts{};

    If the object itself has no comment, the comment of ``<name>_fn`` is used, since ``inline constexpr struct foo_fn
    {...} foo;`` attaches it to the struct.

Names that ``kamping::v2`` re-exports from mpi-core (``using mpi::experimental::send_buffer;`` in a v2 header) are
shown under their v2 name: a page can document ``kamping::v2::send_buffer``, and signatures and code write
``send_buffer`` / ``kamping::v2::send_buffer`` instead of ``mpi::experimental::send_buffer``.

``@ref`` and auto-links in comments become C++ cross-references; links to entities without a docs entry are plain
code, without a warning.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import directives
from docutils.statemachine import StringList
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.domains.cpp import CPPClassObject, CPPConceptObject, CPPFunctionObject, CPPMemberObject, CPPTypeObject, CPPNamespacePopObject, CPPNamespacePushObject
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective

logger = logging.getLogger(__name__)


# --- Doxygen XML index -----------------------------------------------------------------------------------------------


class _Index:
    def __init__(self, xml_dir: Path):
        self.by_name: dict[str, ET.Element] = {}
        self.name_by_id: dict[str, str] = {}
        self.ids: set[str] = set()
        self.functions: dict[str, list[ET.Element]] = {}
        self.members_by_id: dict[str, list[ET.Element]] = {}
        for path in sorted(xml_dir.glob("*.xml")):
            if path.name == "index.xml":
                continue
            for compound in ET.parse(path).getroot().iter("compounddef"):
                cname = compound.findtext("compoundname") or ""
                kind = compound.get("kind")
                self.by_name.setdefault(cname, compound)
                self.ids.add(compound.get("id", ""))
                if kind != "file" and kind != "dir" and kind != "page":
                    self.name_by_id[compound.get("id", "")] = cname
                for member in compound.iter("memberdef"):
                    name = member.findtext("name") or ""
                    qualified = name if kind == "file" else f"{cname}::{name}"
                    self.by_name.setdefault(qualified, member)
                    if member.get("kind") == "function" and kind != "file":
                        self.functions.setdefault(qualified, []).append(member)
                    self.name_by_id.setdefault(member.get("id", ""), qualified)
                    self.ids.add(member.get("id", ""))
                    self.members_by_id.setdefault(member.get("id", ""), []).append(member)


_indices: dict[str, _Index] = {}
_reexports: dict[str, dict[str, str]] = {}


def reexports(env) -> dict[str, str]:
    """Names re-exported by ``kamping::v2`` from ``mpi::experimental``, mapped to the v2 header that does it."""
    include = env.config.kamping_doxygen_v2_include
    if include not in _reexports:
        found: dict[str, str] = {}
        for header in sorted(Path(include).rglob("*.hpp")):
            for name in re.findall(r"^\s*using mpi::experimental::(\w+);", header.read_text(), re.MULTILINE):
                found.setdefault(name, header.relative_to(include).as_posix())
        _reexports[include] = found
    return _reexports[include]


def link_target(name: str, env) -> str:
    """The name under which an entity is declared in the docs: re-exported core names as ``kamping::v2::X``, except
    traits, which are documented (and specialized) as ``mpi::experimental::X_traits``; template arguments in
    Doxygen's spelling (``iresult< Buf >``) without spaces."""
    name = re.sub(r"\s*([<>,])\s*", lambda m: m[1] + (" " if m[1] == "," else ""), name).rstrip()
    # the re-exported entity, or the class a member belongs to: mpi::experimental::request_vector::push_back
    match = re.match(r"(?:kamping::v2|mpi::experimental)::(\w+)(.*)", name)
    if match and match[1] in reexports(env):
        namespace = "mpi::experimental" if match[1].endswith("_traits") else "kamping::v2"
        name = f"{namespace}::{match[1]}{match[2]}"
    # a member of a partial specialization such as iresult<Buf>::wait: Sphinx declares partial specializations under
    # the name of the template, so the reference names it without arguments
    return re.sub(r"(\w)<\w+(?:, \w+)*>::", r"\1::", name)


def v2_names(text: str, env, namespace: str = "") -> str:
    """Spell re-exported core names as ``kamping::v2::X``; inside ``kamping::v2`` drop that qualification."""
    names = reexports(env)
    # traits are specialized under their own name: `template <> struct mpi::experimental::buffer_traits<T>` stays
    renamed = lambda m: f"kamping::v2::{m[1]}" if m[1] in names and not m[1].endswith("_traits") else m[0]
    text = re.sub(r"\bmpi::experimental::(\w+)", renamed, text)
    if namespace == "kamping::v2" or namespace.startswith("kamping::v2::"):
        text = re.sub(r"(?<![\w:])kamping::v2::", "", text)
    return text


def _index(env) -> _Index:
    xml_dir = env.config.kamping_doxygen_xml
    if xml_dir not in _indices:
        _indices[xml_dir] = _Index(Path(xml_dir))
    return _indices[xml_dir]


def _documented(element: ET.Element | None) -> bool:
    if element is None:
        return False
    return any("".join(element.find(s).itertext()).strip() for s in ("briefdescription", "detaileddescription"))


def split_qualified(name: str) -> tuple[str, str]:
    """``("a::b", "c<x::y>")`` for ``"a::b::c<x::y>"``: splits at the last ``::`` outside template arguments."""
    depth = 0
    for i in range(len(name) - 1, 0, -1):
        if name[i] == ">":
            depth += 1
        elif name[i] == "<":
            depth -= 1
        elif depth == 0 and name[i - 1 : i + 1] == "::":
            return name[: i - 1], name[i + 1 :]
    return "", name


def _text(element: ET.Element | None) -> str:
    return " ".join("".join(element.itertext()).split()) if element is not None else ""


def template_header(element: ET.Element) -> str:
    """``template <...> requires ...`` of a templated entity, or ``""``."""
    tparams = element.find("templateparamlist")
    if tparams is None:
        return ""
    params = []
    for param in tparams.findall("param"):
        text = " ".join(filter(None, [_text(param.find("type")), param.findtext("declname")]))
        if param.find("defval") is not None:
            text += " = " + _text(param.find("defval"))
        params.append(text)
    header = f"template <{', '.join(params)}> "
    if element.find("requiresclause") is not None:
        header += f"requires {_text(element.find('requiresclause'))} "
    return header


def function_signature(member: ET.Element) -> str:
    """A C++ declaration of a function memberdef, unqualified (declare it inside its namespace or class)."""
    specifiers = " ".join(
        word for word, attribute in (("static", "static"), ("explicit", "explicit")) if member.get(attribute) == "yes"
    )
    declaration = " ".join(filter(None, [specifiers, _text(member.find("type")), member.findtext("name")]))
    return template_header(member) + declaration + (member.findtext("argsstring") or "")


def _header(element: ET.Element) -> str | None:
    """The header to include for an entity, as users write it."""
    include = element.find("includes")
    if include is not None and include.text:
        return include.text
    location = element.find("location")
    if location is not None:
        match = re.search(r"/include/(.+)$", location.get("file", ""))
        if match:
            return match.group(1)
    return None


# --- Doxygen description XML -> docutils nodes -----------------------------------------------------------------------

_BLOCK_TAGS = {"itemizedlist", "orderedlist", "programlisting", "simplesect", "parameterlist"}


class _Converter:
    """Converts Doxygen description XML. With ``collect`` set, parameter lists and return values are not rendered in
    place but collected into it (``{"params": {kind: [(name, nodes)]}, "returns": [nodes]}``), so that the
    overloads of a function can share them."""

    # inline code is also looked up in these namespaces, e.g. `resize` and `views::resize`
    _SEARCH = ("kamping::v2::views", "kamping::v2", "mpi::experimental", "dstl")

    def __init__(self, env, collect: dict | None = None, scope: str = "", element: ET.Element | None = None):
        self.env = env
        self.index = _index(env)
        self.collect = collect
        self.scope = scope  # namespace of the documented entity, searched first
        # names that are not links in this comment: the parameters of the documented entity
        self.local_names = set()
        if element is not None:
            for param in element.iter("param"):
                for tag in ("declname", "defname"):
                    if param.findtext(tag):
                        self.local_names.add(param.findtext(tag))

    def _lookup_code(self, text: str) -> str | None:
        """The qualified name of the documented entity that inline code such as `resize`, `views::resize` or
        `iresult::wait()` names, or None."""
        match = re.fullmatch(r"([A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)(?:<[^()]*>)?(?:\(\))?", text.strip())
        if not match or match[1] in self.local_names:
            return None
        name = match[1]
        scopes = []
        scope = self.scope
        while scope:
            scopes.append(scope)
            scope = split_qualified(scope)[0]
        for scope in [*scopes, *self._SEARCH, ""]:
            candidate = f"{scope}::{name}" if scope else name
            if candidate in self.index.by_name or candidate in self.index.functions:
                return candidate
            outer, short = split_qualified(candidate)
            if outer == "kamping::v2" and short in reexports(self.env):
                return candidate
        return None

    def _code(self, text: str) -> nodes.Node:
        target = self._lookup_code(text)
        literal = nodes.literal(text=v2_names(text, self.env))
        if target is None:
            return literal
        xref = addnodes.pending_xref(
            "", literal, refdomain="cpp", reftype="any", reftarget=link_target(target, self.env), refexplicit=True
        )
        xref["kamping_doxygen"] = True
        return xref

    def _xref(self, ref: ET.Element) -> nodes.Node:
        text = v2_names("".join(ref.itertext()), self.env)
        target = self.index.name_by_id.get(ref.get("refid", ""))
        if not target:
            return nodes.literal(text=text)
        target = link_target(target, self.env)
        xref = addnodes.pending_xref(
            "", nodes.literal(text=text), refdomain="cpp", reftype="any", reftarget=target, refexplicit=True
        )
        xref["kamping_doxygen"] = True  # unresolved: render as code, without a warning (see _quiet_missing)
        return xref

    def inline(self, element: ET.Element) -> list[nodes.Node]:
        result: list[nodes.Node] = []
        if element.text:
            result.append(nodes.Text(v2_names(element.text, self.env)))
        for child in element:
            if child.tag == "ref":
                result.append(self._xref(child))
            elif child.tag == "computeroutput":
                refs = child.findall("ref")
                if len(refs) == 1 and "".join(child.itertext()) == "".join(refs[0].itertext()):
                    result.append(self._xref(refs[0]))
                else:
                    result.append(self._code("".join(child.itertext())))
            elif child.tag == "emphasis":
                result.append(nodes.emphasis(text="".join(child.itertext())))
            elif child.tag == "bold":
                result.append(nodes.strong(text="".join(child.itertext())))
            elif child.tag == "ulink":
                result.append(nodes.reference(text="".join(child.itertext()), refuri=child.get("url", "")))
            elif child.tag not in _BLOCK_TAGS:
                result.extend(self.inline(child))
            if child.tail:
                result.append(nodes.Text(v2_names(child.tail, self.env)))
        return result

    def para(self, para: ET.Element) -> list[nodes.Node]:
        """A Doxygen <para> may contain block elements (lists, code, sections); split it accordingly."""
        out: list[nodes.Node] = []
        inline = self.inline(para)
        if "".join(n.astext() for n in inline).strip():
            out.append(nodes.paragraph("", "", *inline))
        for child in para:
            if child.tag in ("itemizedlist", "orderedlist"):
                lst = nodes.bullet_list() if child.tag == "itemizedlist" else nodes.enumerated_list()
                for item in child.findall("listitem"):
                    lst += nodes.list_item("", *self.paras(item))
                out.append(lst)
            elif child.tag == "programlisting":
                if child.get("filename") != ".signature":  # signatures are rendered by doxygenview
                    code = v2_names(code_text(child), self.env)
                    out.append(nodes.literal_block(code, code, language="cpp"))
            elif child.tag == "simplesect":
                if self.collect is not None and child.get("kind") == "return":
                    self.collect["returns"].append(self.paras(child))
                else:
                    out.extend(self.simplesect(child))
            elif child.tag == "parameterlist":
                if self.collect is not None:
                    for name, desc in self.parameter_items(child):
                        self.collect["params"].setdefault(child.get("kind", "param"), []).append((name, desc))
                else:
                    out.append(self.parameterlist(child))
        return out

    def paras(self, element: ET.Element) -> list[nodes.Node]:
        return [n for p in element.findall("para") for n in self.para(p)]

    def simplesect(self, sect: ET.Element) -> list[nodes.Node]:
        kind = sect.get("kind", "")
        body = self.paras(sect)
        if kind == "return":
            return [_field_list("Returns", body)]
        if kind in ("note", "warning"):
            admonition = nodes.note() if kind == "note" else nodes.warning()
            admonition.extend(body)
            return [admonition]
        if kind in ("pre", "post"):
            admonition = nodes.admonition()
            admonition += nodes.title(text="Precondition" if kind == "pre" else "Postcondition")
            admonition.extend(body)
            return [admonition]
        if kind == "see":
            return [_field_list("See also", body)]
        return body

    def parameter_items(self, plist: ET.Element) -> list[tuple[str, list[nodes.Node]]]:
        return [
            (", ".join("".join(n.itertext()) for n in item.iter("parametername")),
             self.paras(item.find("parameterdescription")))
            for item in plist.findall("parameteritem")
        ]

    def parameterlist(self, plist: ET.Element) -> nodes.field_list:
        return parameter_field(plist.get("kind", "param"), self.parameter_items(plist))

    def description(self, element: ET.Element) -> list[nodes.Node]:
        out: list[nodes.Node] = []
        for section in ("briefdescription", "detaileddescription"):
            desc = element.find(section)
            if desc is not None:
                out.extend(self.paras(desc))
        return out


def code_text(listing: ET.Element) -> str:
    lines = []
    for codeline in listing.iter("codeline"):
        text = ""
        for part in codeline.iter():
            if part is codeline:
                continue
            text += " " if part.tag == "sp" else (part.text or "")
            text += part.tail or ""
        lines.append(text)
    return "\n".join(lines)


_PARAMETER_LABELS = {"param": "Parameters", "templateparam": "Template parameters", "exception": "Throws"}


def parameter_field(kind: str, items: list[tuple[str, list[nodes.Node]]]) -> nodes.field_list:
    lst = nodes.bullet_list()
    for name, desc in items:
        first = [nodes.literal(text=name), nodes.Text(" – ")]
        if desc and isinstance(desc[0], nodes.paragraph):
            first.extend(desc[0].children)
            desc = desc[1:]
        lst += nodes.list_item("", nodes.paragraph("", "", *first), *desc)
    return _field_list(_PARAMETER_LABELS.get(kind, "Parameters"), [lst])


def overload_label(numbers: list[int]) -> str:
    """(1), (1, 2), (1-3) as on cppreference."""
    if len(numbers) > 2 and numbers == list(range(numbers[0], numbers[-1] + 1)):
        return f"({numbers[0]}-{numbers[-1]})"
    return "(" + ", ".join(map(str, numbers)) + ")"


def _field_list(name: str, body: list[nodes.Node]) -> nodes.field_list:
    return nodes.field_list("", nodes.field("", nodes.field_name(text=name), nodes.field_body("", *body)))


# --- directives ------------------------------------------------------------------------------------------------------


class _DoxygenObject:
    """Mixin for C++ domain directives whose signature and description come from the Doxygen XML.

    The directive argument is the qualified name of the entity. Signatures are declared inside the entity's
    namespace (as with ``cpp:namespace-push``), so names in them are looked up from there.
    """

    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True  # e.g. "kamping::v2::iresult< SBuf, RBuf >"
    has_content = True

    domain_directive: str  # the C++ domain directive this behaves as, e.g. "cpp:function"

    def signatures(self, element: ET.Element, short_name: str) -> list[str]:
        raise NotImplementedError

    def preamble(self, element: ET.Element) -> list[nodes.Node]:
        return []

    reexport_header: str | None = None
    preset: ET.Element | None = None  # the entity, for members rendered by DoxygenClass
    display_namespace: str | None = None  # namespace whose qualification is dropped in signatures (default: own)
    show_header = True
    is_member = False  # a class member: its name is not split at "::" (e.g. "operator mpi::experimental::comm_view")

    def core_name(self, qualified: str) -> str:
        """The mpi-core name behind a re-exported ``kamping::v2::X``, otherwise ``qualified`` itself."""
        namespace, short = split_qualified(qualified)
        if namespace == "kamping::v2" and short in reexports(self.env):
            self.reexport_header = reexports(self.env)[short]
            return f"mpi::experimental::{short}"
        return qualified

    def lookup(self, qualified: str) -> ET.Element | None:
        if self.preset is not None:
            return self.preset
        index = _index(self.env)
        qualified = self.core_name(qualified)
        element = index.by_name.get(qualified)
        self._declaration = element  # the entity itself, even if its comment is taken from elsewhere
        # `inline constexpr struct foo_fn {...} foo;` and `struct foo_t {}; inline constexpr foo_t foo{};` document
        # the struct
        for suffix in ("_fn", "_t"):
            if not _documented(element) and _documented(index.by_name.get(qualified + suffix)):
                element = index.by_name[qualified + suffix]
        return element

    def run(self) -> list[nodes.Node]:
        qualified = self.arguments[0]
        element = self.lookup(qualified)
        if element is None:
            logger.warning("no Doxygen entity named %r", qualified, location=self.get_location())
            return []
        namespace, short_name = ("", qualified) if self.is_member else split_qualified(qualified)
        signatures = self.signatures(element, short_name)
        if not signatures:
            logger.warning("%r has no signature to show", qualified, location=self.get_location())
            return []

        self._element = element
        self._namespace = namespace
        self.name = self.domain_directive
        display = namespace if self.display_namespace is None else self.display_namespace
        self.arguments = ["\n".join(v2_names(sig, self.env, display) for sig in signatures)]
        result = self._scoped(CPPNamespacePushObject, [namespace]) if namespace else []
        objects = super().run()
        # one entry per entity in the page outline, not one per overload
        for desc in objects:
            if isinstance(desc, addnodes.desc):
                for signature in list(desc.findall(addnodes.desc_signature))[1:]:
                    signature["_toc_parts"], signature["_toc_name"] = (), ""
        result += objects
        if namespace:
            result += self._scoped(CPPNamespacePopObject, [])
        return result

    def _scoped(self, directive, arguments) -> list[nodes.Node]:
        return directive(
            directive.__name__, arguments, {}, StringList(), self.lineno, self.content_offset, "", self.state,
            self.state_machine,
        ).run()

    def transform_content(self, content_node) -> None:
        scope = self._namespace if not self.is_member else (self.display_namespace or "")
        converter = _Converter(self.env, scope=scope, element=self._element)
        prepend = self.preamble(self._element) + converter.description(self._element)
        header = self.reexport_header or _header(self._element)
        if header and self.show_header:
            prepend.append(nodes.paragraph("", "Defined in header ", nodes.literal(text=f"<{header}>")))
        content_node[0:0] = prepend  # slice assignment sets the parent of the inserted nodes
        self.after_description(content_node)

    def after_description(self, content_node) -> None:
        pass


class DoxygenView(_DoxygenObject, CPPFunctionObject):
    domain_directive = "cpp:function"
    option_spec = dict(CPPFunctionObject.option_spec)

    def signatures(self, element: ET.Element, short_name: str) -> list[str]:
        listing = next((p for p in element.iter("programlisting") if p.get("filename") == ".signature"), None)
        if listing is None:
            return []
        # one declaration per line; indented lines continue the previous one
        signatures: list[str] = []
        for line in code_text(listing).splitlines():
            if not line.strip():
                continue
            if line[0].isspace() and signatures:
                signatures[-1] += " " + line.strip()
            else:
                signatures.append(line.strip())
        return [s.rstrip(";") for s in signatures]


class DoxygenConcept(_DoxygenObject, CPPConceptObject):
    domain_directive = "cpp:concept"
    option_spec = dict(CPPConceptObject.option_spec)

    def signatures(self, element: ET.Element, short_name: str) -> list[str]:
        params = ", ".join(_text(p.find("type")) for p in element.find("templateparamlist").findall("param"))
        return [f"template <{params}> {short_name}"]

    def preamble(self, element: ET.Element) -> list[nodes.Node]:
        listing = element.find("conceptparts/codepart/programlisting")
        if listing is not None:  # Doxygen >= 1.13: the definition as written
            code = code_text(listing)
        elif element.find("initializer") is not None:  # older Doxygen: the definition with qualified names
            code = "".join(element.find("initializer").itertext())
        else:
            return []
        code = v2_names(code, self.env, self._namespace)
        return [nodes.literal_block(code, code, language="cpp")]


class DoxygenVariable(_DoxygenObject, CPPMemberObject):
    """Shows the declaration without ``inline constexpr``, in v2 names."""

    domain_directive = "cpp:var"
    option_spec = dict(CPPMemberObject.option_spec)

    def signatures(self, element: ET.Element, short_name: str) -> list[str]:
        declaration = getattr(self, "_declaration", None)
        if declaration is None or declaration.get("kind") != "variable":
            declaration = element
        declared = re.sub(r"\b(inline|constexpr|static)\s+", "", _text(declaration.find("type")))
        return [f"{declared} {short_name}"]


class _DoxygenFunctionGroup(_DoxygenObject, CPPFunctionObject):
    """All overloads of a function in one entry, run by DoxygenOverloads: the signatures are numbered, and below
    them each description is labelled with the overloads it applies to; parameters and return values are shared."""

    domain_directive = "cpp:function"
    option_spec = dict(CPPFunctionObject.option_spec)
    members: list[ET.Element]

    def lookup(self, qualified: str) -> ET.Element | None:
        return self.members[0]

    def core_name(self, qualified: str) -> str:
        return qualified

    def signatures(self, element: ET.Element, short_name: str) -> list[str]:
        return [function_signature(m) for m in self.members]

    def run(self) -> list[nodes.Node]:
        result = super().run()
        if len(self.members) > 1:
            signatures = [sig for desc in result if isinstance(desc, addnodes.desc) for sig in desc.findall(addnodes.desc_signature)]
            for number, signature in enumerate(signatures, start=1):
                signature += nodes.inline("", f"({number})", classes=["overload-number"])
        return result

    def transform_content(self, content_node) -> None:
        collect: dict = {"params": {}, "returns": []}
        scope = self._namespace if not self.is_member else (self.display_namespace or "")
        converter = _Converter(self.env, collect, scope=scope, element=self.members[0])
        for member in self.members[1:]:
            converter.local_names |= _Converter(self.env, element=member).local_names
        bodies = []
        for member in self.members:
            returns_before = len(collect["returns"])
            body = converter.description(member)
            returned = collect["returns"][returns_before:]
            bodies.append((body, returned[0] if returned else []))

        out: list[nodes.Node] = []
        # consecutive overloads with the same description share one label
        groups: list[tuple[list[int], list[nodes.Node]]] = []
        for number, (body, _) in enumerate(bodies, start=1):
            if groups and nodes.Element("", *groups[-1][1]).astext() == nodes.Element("", *body).astext():
                groups[-1][0].append(number)
            else:
                groups.append(([number], body))
        for numbers, body in groups:
            if len(groups) > 1:
                label = [nodes.strong(text=overload_label(numbers)), nodes.Text(" ")]
                if body and isinstance(body[0], nodes.paragraph):
                    body[0][0:0] = label
                else:
                    body.insert(0, nodes.paragraph("", "", *label))
            out.extend(body)

        for kind in ("templateparam", "param", "exception"):
            items, seen = [], set()
            for name, desc in collect["params"].get(kind, []):
                if name not in seen:
                    seen.add(name)
                    items.append((name, desc))
            if items:
                out.append(parameter_field(kind, items))

        returns: list[tuple[list[int], list[nodes.Node]]] = []
        for number, (_, returned) in enumerate(bodies, start=1):
            if not returned:
                continue
            match = next((r for r in returns if nodes.Element("", *r[1]).astext() == nodes.Element("", *returned).astext()), None)
            if match:
                match[0].append(number)
            else:
                returns.append(([number], returned))
        if len(returns) == 1:
            out.append(_field_list("Returns", returns[0][1]))
        elif returns:
            lst = nodes.bullet_list()
            for numbers, returned in returns:
                if returned and isinstance(returned[0], nodes.paragraph):
                    returned[0][0:0] = [nodes.strong(text=overload_label(numbers)), nodes.Text(" ")]
                lst += nodes.list_item("", *returned)
            out.append(_field_list("Returns", [lst]))

        header = self.reexport_header or _header(self.members[0])
        if header and self.show_header:
            out.append(nodes.paragraph("", "Defined in header ", nodes.literal(text=f"<{header}>")))
        content_node[0:0] = out


class DoxygenOverloads(SphinxDirective):
    """All overloads of the function named by the argument, in source order."""

    required_arguments = 1
    has_content = True

    def run(self) -> list[nodes.Node]:
        qualified = self.arguments[0]
        functions = {
            name: [m for m in overloads if not (m.findtext("argsstring") or "").endswith("=delete")]
            for name, overloads in _index(self.env).functions.items()
        }
        namespace, short = split_qualified(qualified)
        reexport_header = None
        if namespace == "kamping::v2" and short in reexports(self.env):
            reexport_header = reexports(self.env)[short]
            members = functions.get(f"mpi::experimental::{short}", [])
        else:
            members = functions.get(qualified, [])
        if not members:
            logger.warning("no Doxygen function named %r", qualified, location=self.get_location())
            return []
        directive = _DoxygenFunctionGroup(
            "doxygenoverloads", [qualified], {}, self.content, self.lineno, self.content_offset, self.block_text,
            self.state, self.state_machine,
        )
        directive.members = sorted(
            members, key=lambda m: (m.find("location").get("file"), int(m.find("location").get("line")))
        )
        directive.reexport_header = reexport_header
        return directive.run()


class DoxygenType(_DoxygenObject, CPPTypeObject):
    """A type alias (``doxygentypedef``), also used for the member types rendered by DoxygenClass."""

    domain_directive = "cpp:type"
    option_spec = dict(CPPTypeObject.option_spec)

    def signatures(self, element: ET.Element, short_name: str) -> list[str]:
        aliased = _text(element.find("type"))
        return [f"{short_name} = {aliased}" if aliased else short_name]


_MEMBER_SECTIONS = ("public-type", "public-func", "public-static-func", "public-attrib", "public-static-attrib")


class DoxygenClass(_DoxygenObject, CPPClassObject):
    """The class comment and, with ``:members:``, its
    documented public members, including inherited ones (Doxygen's ``INLINE_INHERITED_MEMB``). Overloaded member
    functions share one numbered entry, as with ``doxygenoverloads``. Members named ``mpi_*`` implement the buffer
    and handle protocol for the library and are left out."""

    domain_directive = "cpp:class"
    option_spec = {**CPPClassObject.option_spec, "members": directives.flag}

    def signatures(self, element: ET.Element, short_name: str) -> list[str]:
        self.domain_directive = "cpp:struct" if element.get("kind") == "struct" else "cpp:class"
        return [template_header(element) + short_name]

    def after_description(self, content_node) -> None:
        if "members" not in self.options:
            return
        functions: dict[str, list[ET.Element]] = {}
        entries: list[tuple[str, object]] = []
        for section in self._element.findall("sectiondef"):
            if section.get("kind") not in _MEMBER_SECTIONS:
                continue
            for member in section.findall("memberdef"):
                # an inherited member template can lack its template header; the base class has the full declaration
                versions = _index(self.env).members_by_id.get(member.get("id", ""), [member])
                member = next((m for m in versions if m.find("templateparamlist") is not None), member)
                kind, name = member.get("kind"), member.findtext("name")
                # mpi_handle(), mpi_ptr(), ...: the buffer and handle protocol, used by the library, not by users
                if not _documented(member) or name.startswith("mpi_"):
                    continue
                if kind == "function":
                    if name not in functions:
                        functions[name] = []
                        entries.append(("function", name))
                    # inherited members can be listed twice (once from the base, once inlined), with or without their
                    # requires clause
                    key = lambda m: (_text(m.find("type")), m.findtext("argsstring"))
                    if key(member) not in map(key, functions[name]):
                        functions[name].append(member)
                elif kind in ("variable", "typedef"):
                    entries.append((kind, member))
        for kind, entry in entries:
            if kind == "function":
                directive = self._member(_DoxygenFunctionGroup, entry)
                directive.members = functions[entry]
            else:
                directive = self._member(DoxygenVariable if kind == "variable" else DoxygenType, entry.findtext("name"))
                directive.preset = entry
            content_node.extend(directive.run())

    def _member(self, cls, name: str):
        directive = cls(
            "doxygenmember", [name], {}, StringList(), self.lineno, self.content_offset, "", self.state,
            self.state_machine,
        )
        directive.display_namespace = self._namespace
        directive.show_header = False
        directive.is_member = True
        return directive


def _unlinked(app: Sphinx, env, node, contnode) -> nodes.Node | None:
    """References generated from the Doxygen XML to entities without a docs entry render as plain text, without a
    warning: ours (marked), and those to Doxygen ids (e.g. source pages). Names in signatures (``cpp:identifier``) link
    if they are documented and are plain text otherwise. References written in the pages still warn."""
    if node.get("kamping_doxygen"):
        logger.verbose("kamping_doxygen: no docs entry for %s", node.get("reftarget"), location=node)  # sphinx-build -v
        return contnode
    if node.get("refdomain") == "cpp" and node.get("reftype") == "identifier":
        return contnode
    target = node.get("reftarget", "")
    if node.get("reftype") == "ref" and (target in _index(env).ids or target.endswith("_source")):
        return contnode
    return None


def setup(app: Sphinx) -> dict:
    app.add_config_value("kamping_doxygen_xml", "", "env")
    app.add_config_value("kamping_doxygen_v2_include", "", "env")
    app.add_directive("doxygenview", DoxygenView)
    app.add_directive("doxygenoverloads", DoxygenOverloads)
    app.add_directive("doxygenconcept", DoxygenConcept)
    app.add_directive("doxygenvariable", DoxygenVariable)
    app.add_directive("doxygenclass", DoxygenClass)
    app.add_directive("doxygenstruct", DoxygenClass)
    app.add_directive("doxygentypedef", DoxygenType)
    app.connect("missing-reference", _unlinked)
    return {"version": "0.2", "parallel_read_safe": True, "parallel_write_safe": True}
