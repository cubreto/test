"""Heuristic COBOL / JCL / copybook extractor -> KDM-lite graph.

This is intentionally a *heuristic* extractor (regex/keyword based), not a
full COBOL grammar. Its job in the sandbox is to demonstrate the typed,
evidence-backed graph. In the TCS environment this whole module is replaced
by the MasterCraft inventory export; the model.py / loader / spec layers stay.
Every fact it emits is therefore stamped extraction_method="heuristic-cobol-v0"
with confidence 0.7 so it is never confused with a deterministic MasterCraft fact.
"""
from __future__ import annotations

import re

from model import Graph

METHOD = "heuristic-cobol-v0"
CONF = 0.7

# --- regexes (case-insensitive, applied to cleaned source) -----------------
RE_PROGRAM_ID = re.compile(r"PROGRAM-ID\.\s+([A-Z0-9-]+)", re.I)
RE_CALL_LIT = re.compile(r"\bCALL\s+['\"]([A-Z0-9-]+)['\"]", re.I)
RE_COPY = re.compile(r"\bCOPY\s+([A-Z0-9-]+)", re.I)
RE_SELECT = re.compile(r"\bSELECT\s+([A-Z0-9-]+)\s+ASSIGN\s+TO\s+([A-Z0-9-]+)", re.I)
RE_FD = re.compile(r"\bFD\s+([A-Z0-9-]+)", re.I)
RE_01 = re.compile(r"^\s*01\s+([A-Z0-9-]+)", re.I)
# Lookbehind avoids matching the verb inside END-READ / END-WRITE / REWRITE.
RE_READ = re.compile(r"(?<![A-Z0-9-])READ\s+([A-Z0-9-]+)", re.I)
RE_WRITE = re.compile(r"(?<![A-Z0-9-])WRITE\s+([A-Z0-9-]+)", re.I)
RE_REWRITE = re.compile(r"(?<![A-Z0-9-])REWRITE\s+([A-Z0-9-]+)", re.I)
RE_SQL = re.compile(r"EXEC\s+SQL(.*?)END-EXEC", re.I | re.S)
RE_SQL_FROM = re.compile(r"\bFROM\s+([A-Z0-9_]+)", re.I)
RE_SQL_INSERT = re.compile(r"\bINSERT\s+INTO\s+([A-Z0-9_]+)", re.I)
RE_SQL_UPDATE = re.compile(r"\bUPDATE\s+([A-Z0-9_]+)", re.I)
RE_SQL_DELETE = re.compile(r"\bDELETE\s+FROM\s+([A-Z0-9_]+)", re.I)
RE_DISPLAY = re.compile(r"\bDISPLAY\b", re.I)
RE_ACCEPT = re.compile(r"\bACCEPT\b", re.I)

# JCL
RE_JCL_JOB = re.compile(r"^//(\S+)\s+JOB\b", re.I)
RE_JCL_EXEC_PGM = re.compile(r"\bEXEC\s+PGM=([A-Z0-9]+)", re.I)


def _clean(text: str):
    """Strip COBOL comment lines / sequence area; return (cleaned, offset->line map)."""
    parts, line_starts, pos = [], [], 0
    for i, raw in enumerate(text.splitlines(), start=1):
        if len(raw) > 6 and raw[6] in "*/":          # fixed-format comment
            piece = ""
        else:
            piece = raw[6:] if len(raw) >= 6 else raw  # drop sequence area cols 1-6
            if "*>" in piece:                          # inline free-form comment
                piece = piece[: piece.index("*>")]
        piece += "\n"
        line_starts.append((pos, i))
        parts.append(piece)
        pos += len(piece)
    return "".join(parts), line_starts


def _lineno(line_starts, offset: int) -> int:
    ln = line_starts[0][1] if line_starts else 1
    for start, i in line_starts:
        if start <= offset:
            ln = i
        else:
            break
    return ln


def parse_copybook(path: str, name: str, text: str):
    """Returns (Graph fragment via callback, primary_record_name)."""
    g = Graph()
    cleaned, lm = _clean(text)
    fields, record_name = [], None
    for raw in cleaned.splitlines():
        m01 = RE_01.match(raw)
        if m01 and record_name is None:
            record_name = m01.group(1).upper()
        mf = re.match(r"\s*\d\d\s+([A-Z0-9-]+)\s+PIC", raw, re.I)
        if mf:
            fields.append(mf.group(1).upper())
    g.add_node("Copybook", name, fields=fields, record=record_name,
               source_artifact=path, source_lines=[1], extraction_method=METHOD,
               confidence=CONF)
    return g, record_name


def parse_cobol(path: str, text: str, copybook_records: dict[str, str]) -> Graph:
    g = Graph()
    cleaned, lm = _clean(text)
    lines = cleaned.splitlines()

    m = RE_PROGRAM_ID.search(cleaned)
    prog = m.group(1).upper() if m else path.rsplit("/", 1)[-1].rsplit(".", 1)[0].upper()
    pid = g.add_node("Program", prog, language="COBOL", source_artifact=path,
                     source_lines=[_lineno(lm, m.start()) if m else 1],
                     extraction_method=METHOD, confidence=CONF)

    # --- data stores from SELECT ... ASSIGN TO ----------------------------
    files: set[str] = set()
    for mm in RE_SELECT.finditer(cleaned):
        logical, physical = mm.group(1).upper(), mm.group(2).upper()
        files.add(logical)
        g.add_node("DataStore", logical, kind="file", physical_name=physical,
                   source_artifact=path, source_lines=[_lineno(lm, mm.start())],
                   extraction_method=METHOD, confidence=CONF)

    # --- FD blocks: map record name -> file (resolve COPY via copybook) ----
    record_to_file: dict[str, str] = {}
    current_fd = None
    for raw in lines:
        mfd = RE_FD.search(raw)
        if mfd:
            current_fd = mfd.group(1).upper()
            continue
        if current_fd:
            m01 = RE_01.match(raw)
            mcopy = RE_COPY.search(raw)
            if m01:
                record_to_file[m01.group(1).upper()] = current_fd
                current_fd = None
            elif mcopy:
                rec = copybook_records.get(mcopy.group(1).upper())
                if rec:
                    record_to_file[rec] = current_fd
                current_fd = None

    # --- copybook usage ----------------------------------------------------
    for mm in RE_COPY.finditer(cleaned):
        cb = mm.group(1).upper()
        cid = g.add_node("Copybook", cb, source_artifact=path,
                         source_lines=[_lineno(lm, mm.start())],
                         extraction_method=METHOD, confidence=CONF)
        g.add_edge(pid, cid, "USES_COPYBOOK", source_artifact=path,
                   source_lines=[_lineno(lm, mm.start())], extraction_method=METHOD,
                   confidence=CONF)

    # --- static calls ------------------------------------------------------
    for mm in RE_CALL_LIT.finditer(cleaned):
        callee = mm.group(1).upper()
        cid = g.add_node("Program", callee, source_artifact=path,
                         source_lines=[_lineno(lm, mm.start())],
                         extraction_method=METHOD, confidence=CONF)
        g.add_edge(pid, cid, "CALLS", source_artifact=path,
                   source_lines=[_lineno(lm, mm.start())], extraction_method=METHOD,
                   confidence=CONF)

    # --- file I/O ----------------------------------------------------------
    def ensure_file(name: str, ln: int) -> str:
        return g.add_node("DataStore", name, kind="file", source_artifact=path,
                          source_lines=[ln], extraction_method=METHOD, confidence=CONF)

    for mm in RE_READ.finditer(cleaned):
        ln = _lineno(lm, mm.start())
        fnode = ensure_file(mm.group(1).upper(), ln)
        g.add_edge(pid, fnode, "READS_FROM", source_artifact=path, source_lines=[ln],
                   extraction_method=METHOD, confidence=CONF)
    for rex in (RE_WRITE, RE_REWRITE):
        for mm in rex.finditer(cleaned):
            ln = _lineno(lm, mm.start())
            operand = mm.group(1).upper()
            target = record_to_file.get(operand, operand)
            fnode = ensure_file(target, ln)
            g.add_edge(pid, fnode, "WRITES_TO", source_artifact=path, source_lines=[ln],
                       extraction_method=METHOD, confidence=CONF)

    # --- embedded SQL ------------------------------------------------------
    for mm in RE_SQL.finditer(cleaned):
        block, ln = mm.group(1), _lineno(lm, mm.start())
        for tm in RE_SQL_FROM.finditer(block):       # SELECT ... FROM t  (read)
            t = tm.group(1).upper()
            tid = g.add_node("DataStore", t, kind="table", source_artifact=path,
                             source_lines=[ln], extraction_method=METHOD, confidence=CONF)
            g.add_edge(pid, tid, "READS_FROM", source_artifact=path, source_lines=[ln],
                       extraction_method=METHOD, confidence=CONF)
        for rex, _ in ((RE_SQL_INSERT, "ins"), (RE_SQL_UPDATE, "upd"), (RE_SQL_DELETE, "del")):
            for tm in rex.finditer(block):
                t = tm.group(1).upper()
                tid = g.add_node("DataStore", t, kind="table", source_artifact=path,
                                 source_lines=[ln], extraction_method=METHOD, confidence=CONF)
                g.add_edge(pid, tid, "WRITES_TO", source_artifact=path, source_lines=[ln],
                           extraction_method=METHOD, confidence=CONF)

    # --- screen (heuristic: interactive DISPLAY + ACCEPT) ------------------
    if RE_DISPLAY.search(cleaned) and RE_ACCEPT.search(cleaned):
        sid = g.add_node("Screen", f"{prog}-SCREEN", source_artifact=path,
                         source_lines=[1], extraction_method=METHOD, confidence=0.5)
        g.add_edge(sid, pid, "SCREEN_CALLS_PROGRAM", source_artifact=path,
                   source_lines=[1], extraction_method=METHOD, confidence=0.5)

    return g


def parse_jcl(path: str, text: str) -> Graph:
    g = Graph()
    job = None
    for i, raw in enumerate(text.splitlines(), start=1):
        if raw.startswith("//*"):
            continue
        mj = RE_JCL_JOB.match(raw)
        if mj:
            job = mj.group(1).upper()
            g.add_node("Job", job, source_artifact=path, source_lines=[i],
                       extraction_method=METHOD, confidence=CONF)
        me = RE_JCL_EXEC_PGM.search(raw)
        if me and job:
            prog = me.group(1).upper()
            jid = f"Job:{job}"
            pid = g.add_node("Program", prog, source_artifact=path, source_lines=[i],
                             extraction_method=METHOD, confidence=CONF)
            g.add_edge(jid, pid, "JOB_RUNS_PROGRAM", source_artifact=path,
                       source_lines=[i], extraction_method=METHOD, confidence=CONF)
    return g
