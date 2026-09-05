import io, re, sys
from collections import Counter

# TABLE-INTEGRITY GATE for docs/gaps.md.
#
# It EXITS NONZERO on a violation. That is deliberate: an earlier version only
# PRINTED, and a `&&` chain sailed straight past a real corruption report and
# pushed the broken row. A check that does not exit is decoration.
#
# ⚠️ NAME THE CONSTRUCT WITH EVERY NUMBER. This script used to print a bare
# "total GAP rows", which matches `^\| ~*GAP-` and therefore INCLUDES
# strikethrough-closed rows -- while `grep -c '^| GAP-'` excludes them. The two
# disagreed 171 vs 168 and the coordinator had been reporting the former as
# "registry rows" for several turns. The arithmetic was right both times; what
# the count RANGED OVER was never stated, so it was never checked.

# ---------------------------------------------------------------------------
# INVARIANT FOR ANY SCRIPT THAT EDITS THIS FILE: POPULATION CONSERVATION.
#
#     A pass that MOVES rows must never create or destroy them. Assert the
#     `^| ~*GAP-` count before and after, and fail if it changed.
#
# ⚠️ WHY THIS IS NOT OBVIOUS, AND WHY PER-ROW CHECKS CANNOT REPLACE IT: on
# 2026-09-02 a pass inserting a missing `Area` cell put the placeholder BEFORE
# the id instead of after it. The rows became `| — | GAP-141 | …`, which no
# longer matches `^| ~*GAP-`, and FOUR ROWS SILENTLY LEFT THE POPULATION --
# 254 -> 250.
#
# EVERY PER-ROW MESSAGE PRINTED SUCCESS, because every individual edit genuinely
# SUCCEEDED. There was no failing item to detect. The rows simply stopped being
# in the population that every later query ranges over, so the relocation that
# followed read as completely clean. The defect is invisible to per-item
# verification BY CONSTRUCTION.
#
# It was caught by a row-total assert that had been added for an unrelated
# reason -- which is the only reason it was there at all.
# ---------------------------------------------------------------------------
lines = io.open('docs/gaps.md', encoding='utf-8').read().split('\n')


def _is_cell_boundary(l, k):
    """Is `l[k]` a `|` that acts as a CELL BOUNDARY -- i.e. not `\\|`-escaped?

    THE ONE DEFINITION OF A CELL BOUNDARY IN THIS FILE, used by both
    `unescaped_pipes` (which counts them) and `row_cells` (which splits on
    them). It was inline in both until 2026-09-02, which made a SECOND copy of
    the rule in the file whose whole lesson is that there must be one -- see
    `unescaped_pipes` below for the incident that lesson came from.

    BOTH halves are load-bearing and neither may be dropped to simplify this:
      * `l[k] == '|'`        the delimiter
      * `k == 0 or ...`      index 0 has no preceding character, so a row's
                             LEADING pipe is always a boundary
      * `l[k - 1] != '\\'`    a `\\|` inside a code span is CONTENT, not a
                             boundary -- GAP-183 reads as 16 cells and has 6
    """
    return l[k] == '|' and (k == 0 or l[k - 1] != '\\')


def unescaped_pipes(l):
    """Count '|' that are NOT escaped as '\\|'.

    ONE implementation, used by every check in this file, and that is
    deliberate rather than tidy.

    On 2026-08-26 TWO PEOPLE INDEPENDENTLY REIMPLEMENTED THIS AND BOTH GOT
    IT WRONG IN THE SAME DIRECTION -- one in awk (`gsub` escaping ate the
    pattern), one in Python (a shell heredoc ate a backslash, leaving the
    regex `\\|`, i.e. the alternation "literal backslash OR empty", which
    strips the BACKSLASH and LEAVES THE PIPE so it then counts as a
    separator). Both produced rows that looked catastrophically malformed
    and were fine -- GAP-183 reads as 16 cells and has 6.

    THEIR TWO WRONG ANSWERS AGREED WITH EACH OTHER, which is far more
    persuasive than one wrong answer, and was nearly filed as a confirmed
    defect in THIS gate. Agreement between instruments that share a blind
    spot is not corroboration. Hence: one function, called everywhere.
    """
    return sum(1 for k in range(len(l)) if _is_cell_boundary(l, k))


rows = []
for l in lines:
    if re.match(r'^\| ~*GAP-', l):
        rid = re.match(r'^\| (~*GAP-[0-9]+)', l).group(1)
        rows.append((rid, unescaped_pipes(l), l))

struck = [r for r in rows if r[0].startswith('~~')]
live = [r for r in rows if not r[0].startswith('~~')]
# NAME THE CONSTRUCT, NOT THE INTENT. This matches CLOSED ANYWHERE IN THE ROW,
# which is NOT the same as "the status cell says CLOSED" -- the label said the
# latter for weeks and was wrong in two ways at once (measured 2026-08-28 on a
# 52-row population): only 43 of the 52 had CLOSED in the LAST cell, and 5 sat
# in 4-column tables that have NO status cell at all, where the sentence is not
# even expressible. That is this file's own four-schema trap, inside the gate
# that warns about the schemas. Keeping the loose match is deliberate -- it is
# the safe direction for a tripwire -- but the label now says what it counts.
closed_word_anywhere = [r for r in live if 'CLOSED' in r[2]]

# SCHEMA COVERAGE, because a status convention is not expressible for a table
# with no status column. The 4-column tables (`ID | File:Line | Tier | Gap`) fold
# status into the Gap cell; the 5-column ones have a real Status cell.
#
# COUNTED, NOT STATED. A convention covering 63% of a file and described in prose
# as covering the file is the green-reads-as-coverage failure -- the same one the
# two doc-vs-code guards each had to close in their own scope statements. Making
# the hole visible is deliberate; closing it (a status column on nine tables) is
# a separate decision nobody has taken. See docs/design/gaps-status-vocabulary.md.
schema_cols, _cur = [], None
for _l in lines:
    if re.match(r'^\| ID', _l):
        _cur = unescaped_pipes(_l) - 1
    elif re.match(r'^\| ~*GAP-', _l):
        schema_cols.append(_cur)
    elif not _l.startswith('|'):
        _cur = None
no_status = sum(1 for c in schema_cols if c == 4)
headerless = sum(1 for c in schema_cols if c is None)

# ---------------------------------------------------------------------------
# CONTROL CHARACTERS -- checked FIRST, because an invisible byte can corrupt the
# row parsing that every other check below depends on.
#
# WHY THIS EXISTS: seven of these accumulated across earlier commits and were
# repaired at `ffc5f25a` -- 5x FORMFEED (0x0c) and 1x BACKSPACE (0x08) in this
# file, 1x VERTICAL TAB (0x0b) in CLAUDE.md. All of them arrived the same way:
# a Windows path or a regex written inside a QUOTED SHELL HEREDOC, where the
# backslash is eaten and `\f` in `C:\Projects\fuel` becomes one formfeed byte.
#
# ⚠️ THE ASYMMETRY IS THE WHOLE REASON THEY ACCUMULATED, AND IT IS THE OPPOSITE
# OF THE INTUITION: `\Projects` and `\.git` raise a SyntaxWarning and SURVIVE AS
# LITERALS, while `\fuel` becomes a formfeed with NO DIAGNOSTIC AT ALL. The
# escapes that warn are the harmless ones; the silent one is the one that
# corrupts. So "be careful with heredocs" is not a usable rule -- nobody was
# careless, and the loud cases trained everyone to expect a warning that the
# damaging case never emits.
#
# ⚠️ CONTEXT IS PRINTED, NOT JUST AN OFFSET, AND THAT IS LOAD-BEARING: the byte
# is INVISIBLE, so a line/column alone tells a reader nothing they can see. What
# identified the original seven was a context dump, not a coordinate.
#
# TAB is allowed per the ruling. CR and LF cannot appear here at all: the file is
# read through universal-newline translation and split on '\n', so any surviving
# control character is genuinely embedded in a cell.
_CTRL_NAMES = {0x08: 'BACKSPACE', 0x09: 'TAB', 0x0b: 'VERTICAL TAB',
               0x0c: 'FORMFEED', 0x1b: 'ESCAPE'}
control_chars = []
for _n, _l in enumerate(lines, 1):
    for _k, _ch in enumerate(_l):
        _o = ord(_ch)
        if _o < 0x20 and _ch != '\t':
            _lo, _hi = max(0, _k - 20), min(len(_l), _k + 21)
            _ctx = _l[_lo:_hi].replace(_ch, '<U+%04X>' % _o)
            control_chars.append(
                'line %d col %d: U+%04X %s -- ...%s...'
                % (_n, _k + 1, _o, _CTRL_NAMES.get(_o, 'CONTROL'), _ctx))

# ---------------------------------------------------------------------------
# CONFLICT MARKERS.
#
# ⚠️ FOUND BY REBASING THIS FILE THREE TIMES IN ONE SESSION: the gate returned
# EXIT 0 on a CONFLICTED docs/gaps.md, every time.
#
# It is structural, not an oversight. Every other check here keys on lines
# matching `^| ~*GAP-` or `^| ID`. Conflict markers start with `<`, `=` and `>`,
# so the row parser does not merely tolerate them -- IT CANNOT SEE THEM. Both
# sides of the conflict are then counted as ordinary rows, the totals go UP, and
# every check still passes. THE FILE IS IN THE MOST BROKEN STATE GIT CAN LEAVE
# IT IN AND THE GATE SAYS CLEAN.
#
# The pre-commit hook cannot save you either: `git add`-ing a conflicted file
# marks it resolved, so the hook runs against exactly this content.
conflict_markers = [
    'line %d: %s' % (n, l[:60])
    for n, l in enumerate(lines, 1)
    if l.startswith('<<<<<<<') or l.startswith('>>>>>>>') or l.rstrip() == '======='
]
print('unresolved conflict markers:',
      conflict_markers if conflict_markers else 'NONE')
print('control characters (excl. tab):',
      control_chars if control_chars else 'NONE')
print()
print('delimiter-count distribution:', dict(Counter(d for _, d, _ in rows)))
odd = [(r, d) for r, d, _ in rows if d not in (5, 6)]
print('rows NOT in {5,6}:', odd if odd else 'NONE')

# A well-formed row ENDS with '|'. This is a SEPARATE failure from the delimiter
# count and the count CANNOT see it: dropping only the trailing pipe takes a row
# from 6 to 5, which is still "valid". The real incident dropped the status cell
# AND the pipe (6 -> 4), so the count caught it by luck of losing two at once.
# First attempt at this check only RELABELLED the row and never failed -- an
# informational check, the exact defect this gate exists to prevent, rebuilt
# inside the gate itself. Validated by sabotage in BOTH directions.
no_pipe = [r for r, _, l in rows if not l.rstrip().endswith('|')]
print('rows with NO TRAILING PIPE:', no_pipe if no_pipe else 'NONE')
print()
print('rows, ALL (regex ^| ~*GAP-, INCLUDES strikethrough-closed): %d' % len(rows))
print('rows, NOT struck through (what `grep -c "^| GAP-"` returns):  %d' % len(live))
print('rows, struck through (closed-by-strikethrough):               %d  %s'
      % (len(struck), [r[0] for r in struck]))
print('rows, live, with the WORD "CLOSED" ANYWHERE in the row:      %d' % len(closed_word_anywhere))
def _pct(n, d):
    """A percentage, never printed alone.

    Every call site below prints `%d of %d (%d%%)`, so the numerator, the
    DENOMINATOR and the ratio travel together. A bare percentage is
    unreadable -- 44% of what? -- and this file's whole subject is counts
    whose population was never stated.
    """
    return round(100.0 * n / d) if d else 0


print('rows OUTSIDE the status convention (4-col, no status cell): %d of %d (%d%%)'
      % (no_status, len(schema_cols), _pct(no_status, len(schema_cols))))
# HEADERLESS ROWS: GAP rows in a table fragment with no `| ID |` header above
# them (a `---` rule then rows). The header check below reports 'headers
# DISAGREEING: NONE' and CANNOT SEE THESE -- there is no header to disagree
# with. Found 2026-08-28 by two schema-counting methods disagreeing 94 vs 92:
# the looser one silently attributed them to a DIFFERENT table's header.
print('rows with NO HEADER above them (schema undetermined):     %d'
      % headerless)
# THE COMBINED FIGURE -- COMPUTED HERE, NOT QUOTED FROM ANYWHERE.
#
# A reader asking "how much of this file does the status convention actually
# cover?" needs BOTH lines above added: a 4-column row has nowhere to PUT a
# status, and a headerless row has no determined schema at all. Printing only
# the first understates the uncovered population by however many rows are
# currently headerless -- which was 20, and those 20 are the entire reason
# this gate spent an unknown window reporting a clean verdict over a
# population it could not see.
#
# WHY THIS IS COMPUTED AND NOT WRITTEN DOWN: the figure was specified to the
# lane as `112 of 253 (44%)`. That was measured, correct, and TRUE ONLY OF
# THE PRE-FIX FILE -- 92 four-column + 20 headerless. Giving those 20 rows a
# 5-column header (their own measured arity) hands each one a real Status
# cell, so they move INTO the convention and the total falls to 92 of 253
# (36%), with the 5-column group going 141 -> 161. The remediation dissolves
# its own reporting target, and a hardcoded 112 would have been a fabricated
# number sitting in a gate whose entire job is to make counts honest.
#
# THE COMPONENTS STAY SEPARATE because they are different defects with
# different owners: a 4-column table needs a ruled decision about adding a
# Status column (nobody has taken it -- docs/design/gaps-status-vocabulary.md),
# whereas a headerless fragment just needs its header. One number would hide
# that half of it is a decision and half of it is a chore.
_uncovered = no_status + headerless
print('rows NOT COVERED by the status convention (both of the above): %d of %d (%d%%)'
      % (_uncovered, len(schema_cols), _pct(_uncovered, len(schema_cols))))
print()
# NOTE: ASCII ONLY below. The first version of this block used an emoji and
# died on cp1252 stdout -- making the gate exit 1 for a PRINTING failure with no
# table violation present. A false red from a gate is worse than no gate: it is
# the failure this project keeps recording, pointed at the instrument itself.
print('!! CLOSE VOCABULARY IS NOT CONTROLLED, so "how many are open" has no single')
print('   answer from this table alone. Strikethrough is the ruled close marker')
print('   (2026-08-28); the count above is the WORD "CLOSED" anywhere in a live')
print('   row, which over-counts -- some of those rows are correctly OPEN and')
print('   merely narrate a closure, and others are honestly PARTIAL and say so')
print('   in their leading phrase. FOURTEEN forms are in use besides the two:')
print('   SHIPPED, FIXED, PARTIALLY CLOSED, RE-OPENED, NOT CLOSED, UNIT A')
print('   CLOSED, LAYER 1 CLOSED, HALF CLOSED, "CLOSED for X ... OPEN for Y",')
print('   "(a) CLOSED; (b) OPEN", WON\'T DO, MISFILED, SUBJECT DELETED,')
print('   "(Was: OPEN ...)". Most carry real information a single token would')
print('   destroy, so the fix is a controlled PREFIX plus free prose, not a')
print('   collapse to two words. Quote the construct with the number.')
print()
print('!! THE HOMED-RESIDUE TEST APPLIES TO CLOSED ROWS ONLY. A closure is')
print('   genuine when its residue is absent or HOMED AT A NAMED ROW -- but')
print('   that is because STRIKETHROUGH IS WHAT TELLS A READER NOT TO LOOK,')
print('   so residue inside a CLOSED row is unreachable. Residue inside an')
print('   OPEN row is exactly where it belongs: AN OPEN ROW IS ITS OWN')
print('   ADDRESS. Applied unbounded on 2026-08-28 the criterion would have')
print('   filed eight rows pointing at eight rows that already point at')
print('   themselves -- the satisfied-non-goal shape, inside a criterion')
print('   written to prevent it.')
print()

# THE TIER CELL IS AUTHORITATIVE; THE SECTION HEADING IS NOT.
#
# Measured 2026-09-02 while headering the three fragments this gate could
# not previously see: all three are CHRONOLOGICAL APPENDS, not tier-sorted.
# The 13-row fragment sitting under `## Tier B` holds 9 B, 3 C and 1 A; the
# 5-row fragment under Tier A holds 3 A and 2 B; the 2-row fragment under
# Tier P holds one P and one TBD.
#
# So a query that groups rows by SECTION BOUNDARY returns a wrong answer
# today, silently. Sorting the rows was ruled against: it would shift every
# row's line number and invalidate citations across the corpus, and it is
# unnecessary because the tier is already in the row. Same precedent as
# rule 4b -- the data outvotes the presentation.
print('!! THE 4-COLUMN TABLES ARE PERMANENTLY EXEMPT FROM THE STATUS')
print('   CONVENTION -- ruled 2026-09-02 on measurement, not convenience.')
print('   They are Tier C subdivided BY CRATE: an INDEX of terse one-line')
print('   capability declines where the gap statement IS the status')
print('   ("Pad Reflect/Replicate modes return \'not yet implemented\'").')
print('   Measured: of 79 such rows, 67 carry NO status language at all, so')
print('   a Status column would mean writing OPEN into 67 cells -- a field')
print('   identical for 85% of its rows. THAT IS NOT INFORMATION, IT IS A')
print('   COLUMN THAT EXISTS TO BE FULL, and it is the same false precision')
print('   the OPEN/<state> ruling rejected, one column over.')
print('   64% honest beats 100% decorative. DO NOT "COMPLETE" THIS.')
print()
print('!! THE TIER CELL (column 3) IS AUTHORITATIVE. The `## Tier X` section')
print('   headings are PRESENTATIONAL and membership in them is CHRONOLOGICAL:')
print('   rows were appended where the file happened to end, not where their')
print('   tier belongs. Measured 2026-09-02: the 13-row block under Tier B')
print('   holds 9 B, 3 C and 1 A. GROUP BY THE CELL, NEVER BY THE SECTION --')
print('   a section-boundary query returns a wrong answer and says nothing.')

# ---------------------------------------------------------------------------
# HEADERS MUST AGREE WITH THE ROWS BENEATH THEM, IN ARITY AND IN NAME.
#
# EVERY CHECK ABOVE EXAMINES ONLY `^| ~*GAP-` ROWS. NOT ONE OF THEM LOOKS AT
# A HEADER, so a table could carry a 4-column header over 5-column rows and
# this gate would print a clean bill of health -- which is exactly what
# happened. The Meta table declared `| ID | Owner | Gap | Status |` while 86
# of its 90 rows carried five columns, so every label rendered one position
# off and each row's Status cell had NOWHERE TO GO. The gate was honest
# about a different question than the one being asked, which is scope, not
# a lie -- but the silence read as coverage.
#
# ARITY IS THE HALF THAT SILENTLY EATS A CELL, and it is the half that was
# actually wrong. A check for the word `Owner` alone would not have caught
# it: the label was renamed twice before anyone noticed a column was
# MISSING.
#
# Also caught here: a PARTIAL rename. Thirteen headers, one renamed, twelve
# left -- which converts a DETECTABLE inconsistency (uniformly wrong, and
# systematic enough that a reader found it) into an undetectable one
# (sample line 24 and the file looks correct; sample any other and it looks
# like the old defect; nothing tells you which you sampled).
header_problems = []
cur = None          # (line_no, text, pipes)
counts = []         # unescaped-pipe counts of GAP rows under `cur`

def close_table():
    if cur is None or not counts:
        return
    modal = Counter(counts).most_common(1)[0][0]
    if cur[2] != modal:
        header_problems.append(
            'line %d: header has %d columns, but %d of its %d rows have %d '
            '-- %s' % (cur[0], cur[2] - 1, counts.count(modal), len(counts),
                       modal - 1, cur[1].strip()))
    cells = [c.strip() for c in cur[1].strip().strip('|').split('|')]
    if 'Owner' in cells:
        header_problems.append(
            'line %d: header names a column `Owner`; the third column is TIER '
            '(ruled 2026-08-26) -- %s' % (cur[0], cur[1].strip()))

for n, l in enumerate(lines, 1):
    if re.match(r'^\| ID\b', l):
        close_table()
        cur, counts = (n, l, unescaped_pipes(l)), []
    elif re.match(r'^\| ~*GAP-', l):
        counts.append(unescaped_pipes(l))
    elif not l.startswith('|'):
        close_table()
        cur, counts = None, []
close_table()

print()
print('headers checked against the rows beneath them: %d'
      % sum(1 for l in lines if re.match(r'^\| ID\b', l)))
print('headers DISAGREEING with their rows:', header_problems if header_problems else 'NONE')

# ---------------------------------------------------------------------------
# PARENT -> CHILD BACKLINKS
#
# A row that homes another row's residue declares `Parent: GAP-NNN`. Nothing
# required the PARENT to cite the child back, so on 2026-09-02 the score was
# 0 of 10 -- every declaration one-way.
#
# ⚠️ WHY THAT IS A DEFECT AND NOT UNTIDINESS, AND IT IS SPECIFIC TO STRUCK
# ROWS: `~~GAP-186~~` says "the generator producing the 42nd accessor is a
# KNOWN OPEN RESIDUAL" and names no row. Its residue IS homed -- at GAP-250 --
# so the strike is legitimate. But STRIKETHROUGH IS WHAT TELLS A READER NOT TO
# LOOK FURTHER, so from the row a reader actually lands on, a HOMED residue and
# an UN-HOMED one are INDISTINGUISHABLE. The work was never lost; the address
# was unreachable from the only place anyone would go.
#
# Three states, not two:
#   un-homed          -> unstrike, or file a row   (GAP-176 -> GAP-258)
#   homed AND cited   -> fine
#   homed but SILENT  -> one clause                (GAP-186 -> GAP-250)
# With only two states, `~~GAP-186~~` reads as un-homed and gets UNSTRUCK,
# re-opening finished work -- and an unstruck row looks like diligence, so
# nothing downstream catches it.
#
# ⚠️ WHY THIS IS A GATE RATHER THAN A RULE: filing a row is one action and
# backlinking is a second action nobody's workflow requires, so the omission is
# the CONVENTION'S DEFAULT. It recurred within one hour, in the person who had
# just been warned about it, while filing GAP-258. Awareness demonstrably does
# not hold; only a check does.
def missing_backlinks_in(src_lines):
    """Rows declaring `Parent: X` whose X does not cite them back.

    A FUNCTION so the retained sabotage below can run the SAME code against a
    fixture. Duplicating the logic for the self-test would validate a copy.
    """
    claims, by_id = {}, {}
    for l in src_lines:
        m = re.match(r'^\| ~*(GAP-[0-9]+)', l)
        if not m:
            continue
        by_id.setdefault(m.group(1), []).append(l)
        for pm in re.finditer(r'Parents?:\s*((?:GAP-[0-9]+[,;\s]*)+)', l):
            for p in re.findall(r'GAP-[0-9]+', pm.group(1)):
                if p != m.group(1):
                    claims.setdefault(p, set()).add(m.group(1))
    out = []
    for p in sorted(claims):
        if p not in by_id:
            out.append('%s is declared as a parent but has NO ROW' % p)
            continue
        joined = ' '.join(by_id[p])
        for c in sorted(claims[p]):
            if c not in joined:
                out.append('%s declares `Parent: %s`, but %s does not cite %s'
                           % (c, p, p, c))
    return out, sum(len(v) for v in claims.values())


# ⚠️ RETAINED SABOTAGE, RUN ON EVERY INVOCATION -- NOT AN AUTHORING-TIME RED.
# This check went green the moment it was written, because the ten real
# backlinks had just been added. A born-red proves a gate discriminated ONCE;
# it says nothing about any later run, and a check that has never been SEEN to
# fire is indistinguishable from one that does nothing. The fixture below is
# missing its backlink BY CONSTRUCTION, so if the detector is ever weakened
# into inertness this foundation check goes red before the real data does.
_FIXTURE_CAUGHT = [
    '| GAP-901 | fixture | C | child row. Parent: GAP-902. |',
    '| ~~GAP-902~~ | fixture | C | parent row that does NOT cite its child. |',
]
_FIXTURE_CLEAN = [
    '| GAP-901 | fixture | C | child row. Parent: GAP-902. |',
    '| ~~GAP-902~~ | fixture | C | parent row citing GAP-901. |',
]
_caught, _ = missing_backlinks_in(_FIXTURE_CAUGHT)
_clean, _ = missing_backlinks_in(_FIXTURE_CLEAN)
_foundation = []
if len(_caught) != 1:
    _foundation.append('detector did NOT flag the sabotaged fixture (got %r)' % _caught)
if _clean:
    _foundation.append('detector flagged the CLEAN fixture (got %r)' % _clean)

missing_backlinks, _n_claims = missing_backlinks_in(lines)

print()
print('parent<-child backlinks checked: %d' % _n_claims)
print('backlink detector self-test (sabotaged fixture must flag, clean must not):',
      'PASS' if not _foundation else _foundation)
print('parents NOT citing their declared child:',
      missing_backlinks if missing_backlinks else 'NONE')

# ---------------------------------------------------------------------------
# ARITY DISSENT -- ANY row disagreeing with its header, not just a MODAL one.
#
# ⚠️ `close_table()` above compares a header to the MODAL row count, so a
# MINORITY of wrong-arity rows never trips it. Eighteen rows disagreed with
# their header while it reported NONE. That is the THIRD time this file has
# carried a check reporting clean over a population it structurally cannot see
# -- and the second time in the same check, after the headerless fragments.
#
# The two directions are DIFFERENT DEFECTS and are reported separately:
#   row has MORE cells than its header  -> the row carries status content the
#                                          header has no column for
#   row has FEWER cells than its header -> a missing cell. This is an ARITY
#                                          defect, NOT a vocabulary violation,
#                                          and the vocabulary check below must
#                                          not index the cell that isn't there.
arity_extra, arity_missing = [], []
_hdr = None
for n, l in enumerate(lines, 1):
    if re.match(r'^\| ID\b', l):
        _hdr = (n, unescaped_pipes(l) - 1)
    elif re.match(r'^\| ~*GAP-', l):
        if _hdr is not None:
            own = unescaped_pipes(l) - 1
            if own > _hdr[1]:
                arity_extra.append('line %d: row has %d cells, header (line %d) has %d'
                                   % (n, own, _hdr[0], _hdr[1]))
            elif own < _hdr[1]:
                arity_missing.append('line %d: row has %d cells, header (line %d) has %d'
                                     % (n, own, _hdr[0], _hdr[1]))
    elif not l.startswith('|'):
        _hdr = None

# ⚠️ A RATCHET, NOT A HARD GATE, AND THE REASON IS NOT LENIENCY. These 18 are
# real, pre-existing and outside the scope of the change that added this check.
# Failing on them would red-gate every commit in the repo until an unrelated
# 18-row edit lands, and a gate that cannot be satisfied by following its own
# advice teaches `--no-verify` -- a worse outcome than the drift it guards.
# Same shape as the prose-hedge allowlist: the baseline MAY ONLY SHRINK.
#
# ⚠️ IF YOU FIX ROWS, LOWER THESE NUMBERS IN THE SAME COMMIT. A baseline left
# above the true count is a gate that has quietly stopped guarding, which is the
# defect this whole file exists to catch.
ARITY_EXTRA_BASELINE = 1
ARITY_MISSING_BASELINE = 0

print()
print('rows with MORE cells than their header (status content, no column): %d (baseline %d)'
      % (len(arity_extra), ARITY_EXTRA_BASELINE))
for _p in arity_extra:
    print('    ' + _p)
print('rows with FEWER cells than their header (ARITY defect, not vocabulary): %d (baseline %d)'
      % (len(arity_missing), ARITY_MISSING_BASELINE))
for _p in arity_missing:
    print('    ' + _p)
arity_regression = []
if len(arity_extra) > ARITY_EXTRA_BASELINE:
    arity_regression.append('MORE-cells rows rose %d -> %d'
                            % (ARITY_EXTRA_BASELINE, len(arity_extra)))
if len(arity_missing) > ARITY_MISSING_BASELINE:
    arity_regression.append('FEWER-cells rows rose %d -> %d'
                            % (ARITY_MISSING_BASELINE, len(arity_missing)))
if len(arity_extra) < ARITY_EXTRA_BASELINE or len(arity_missing) < ARITY_MISSING_BASELINE:
    arity_regression.append(
        'BASELINE IS STALE AND MUST BE LOWERED IN THIS COMMIT: measured %d/%d, '
        'baseline %d/%d -- a baseline above the true count is a gate that has '
        'quietly stopped guarding'
        % (len(arity_extra), len(arity_missing),
           ARITY_EXTRA_BASELINE, ARITY_MISSING_BASELINE))
print('arity ratchet:', arity_regression if arity_regression else 'HOLDING')
# ⚠️ WHY THE `extra` BASELINE IS 1 AND NOT 0 -- ruled 2026-09-02, and it is a
# decision rather than leftover debt.
#
# The single remaining dissent is GAP-099: a worked row, priced (64 literals /
# 3 crates), ranked below C ON MEASUREMENT, and carrying a re-rank trigger. Its
# tier is `—`, for which there is no section, so there is nowhere to relocate
# it to. GAP-048/GAP-079 were folded because their status was a bare `OPEN` the
# Gap text already carried, and nothing was lost. GAP-099 is the opposite kind
# of row: it is precisely the row that OUTGREW the index, and folding its
# status into prose to make this counter reach zero would be FITTING THE DATA
# TO THE INSTRUMENT.
#
# The gate does not need the zero. The stale-baseline arm above already
# prevents the only failure a ratchet has -- it can only shrink, and a baseline
# left above the true count fires. A NAMED, REASONED 1 TELLS A READER
# SOMETHING; A 0 BOUGHT THIS WAY WOULD TELL THEM THE FILE IS UNIFORM, WHICH IS
# FALSE.
if len(arity_extra) == 1 and not arity_regression:
    print('   (the 1 is GAP-099 BY DECISION, not debt: a worked row whose tier')
    print('    has no section. Folding its status to reach 0 would be fitting')
    print('    the data to the instrument. Ruled 2026-09-02.)')

# ---------------------------------------------------------------------------
# STATUS VOCABULARY -- docs/design/gaps-status-vocabulary.md
#
# ⚠️⚠️ THIS BUYS GREPPABILITY, NOT ACCURACY. NOT ONE DEFECT FOUND IN THE
# 2026-08-28 CLOSE-CONVENTION AUDIT WOULD HAVE BEEN CAUGHT BY IT. Every real
# finding that night came from READING A ROW. Whoever cites this file may say
# "N rows carry prefix X" and must NEVER say "N gaps are verified closed" on
# the strength of it. A convention sold as hygiene and read as verification is
# the defect this registry keeps producing -- printed here rather than left in
# the design doc, because the doc is read once and this prints every commit.
#
# SCOPE IS THE ROW'S OWN ARITY, NOT ITS HEADER'S (ruled 2026-09-02). The row is
# where the status lives; 14 rows carry a status cell under a 4-column header
# and a header-keyed rule would exempt rows that are really in scope.
STATUS_PREFIXES = ('CLOSED', 'PARTIAL', 'RE-OPENED', "WON'T DO", 'VOID',
                   'MEASURED', 'OPEN')


def normalise_status(c):
    """Strip, FROM THE LEFT ONLY: whitespace, markdown emphasis, and any
    non-ASCII character -- which is how the `✅` / `⚠️` leads are absorbed.

    ⚠️ THIS IS A JUDGMENT AND IT IS STATED HERE ON PURPOSE. The design doc
    called this test "exact, mechanical, no judgment", and that was literally
    false for 13 of the cells: six lead with an emoji, four are empty, the rest
    open with a backtick or bold. An UNSTATED normalisation inside a check
    billed as having none is the spec being false about itself. The emoji
    carries the same verdict the token does, so it is absorbed rather than
    forbidden -- forbidding it would convert six correct rows into violations
    for no greppability gain.
    """
    i = 0
    while i < len(c) and (c[i].isspace() or c[i] in '*`~' or ord(c[i]) > 127):
        i += 1
    return c[i:]


def status_prefix(c):
    s = normalise_status(c).upper()
    for p in sorted(STATUS_PREFIXES, key=len, reverse=True):
        if s.startswith(p):
            return p
    return None


def row_cells(l):
    out, buf = [], []
    for k, ch in enumerate(l):
        if _is_cell_boundary(l, k):
            out.append(''.join(buf))
            buf = []
        else:
            buf.append(ch)
    out.append(''.join(buf))
    if out and not out[0].strip():
        out = out[1:]
    if out and not out[-1].strip():
        out = out[:-1]
    return [c.strip() for c in out]


def vocab_findings(src_lines):
    """(unrecognised-prefix rows, strikethrough contradictions, distribution).

    A FUNCTION so the retained self-test below runs THIS code against fixtures
    rather than a copy of it.
    """
    bad, contra, dist = [], [], Counter()
    for n, l in enumerate(src_lines, 1):
        _m = re.match(r'^\| (~*GAP-[0-9]+)', l)
        if not _m:
            continue
        _cs = row_cells(l)
        if len(_cs) != 5:      # arity defect, reported above -- NOT here
            continue
        _p = status_prefix(_cs[4])
        dist[_p or '(UNRECOGNISED)'] += 1
        if _p is None:
            bad.append('line %d %s: status does not start with a set member -- %r'
                       % (n, _m.group(1), normalise_status(_cs[4])[:60]))
        elif _m.group(1).startswith('~~') and _p in ('PARTIAL', 'OPEN', 'RE-OPENED'):
        # ⚠️ ASYMMETRIC ON PURPOSE -- DO NOT "COMPLETE" THIS INTO SYMMETRY.
        #   struck + OPEN/PARTIAL -> FLAG. The row is hidden AND has open work,
        #                            so the work is UNREACHABLE: strikethrough
        #                            is what tells a reader not to look.
        #   unstruck + CLOSED     -> do NOT flag. Visible, and says closed.
        #                            Merely untidy; nothing is hidden.
        # Asymmetric damage, asymmetric check.
            contra.append('line %d %s: STRUCK but prefix is %s -- %r'
                          % (n, _m.group(1), _p, normalise_status(_cs[4])[:60]))
    return bad, contra, dist


# ⚠️ RETAINED SELF-TEST, RUN EVERY INVOCATION. The cross-check below is called
# "the ONE genuine accuracy test available here", and on 2026-09-02 it fired on
# REAL DATA (`~~GAP-176~~`) -- the strongest evidence a check can have. But the
# moment that row is dispositioned it goes green forever, and a check nobody has
# seen fire is indistinguishable from one that does nothing. Real-data evidence
# is a SNAPSHOT; this fixture is the STANDING proof.
_FX_BAD = ['| GAP-903 | fx | C | fx | bespoke phrase with no set member |']
_FX_CONTRA = ['| ~~GAP-904~~ | fx | C | fx | **PARTIAL** — work remains |']
_FX_OK = ['| ~~GAP-905~~ | fx | C | fx | **CLOSED** — nothing remains |']
_vb, _vc, _ = vocab_findings(_FX_BAD)
_wb, _wc, _ = vocab_findings(_FX_CONTRA)
_ob, _oc, _ = vocab_findings(_FX_OK)
vocab_foundation = []
if len(_vb) != 1:
    vocab_foundation.append('prefix check did NOT flag an unrecognised lead (%r)' % _vb)
if len(_wc) != 1:
    vocab_foundation.append('cross-check did NOT flag a STRUCK PARTIAL (%r)' % _wc)
if _ob or _oc:
    vocab_foundation.append('a CLEAN struck+CLOSED fixture was flagged (%r %r)' % (_ob, _oc))

bad_prefix, strike_contra, prefix_dist = vocab_findings(lines)

print()
print('vocabulary detector self-test (unrecognised must flag, struck-PARTIAL must '
      'flag, struck-CLOSED must not):',
      'PASS' if not vocab_foundation else vocab_foundation)
print('status-prefix distribution (5-cell rows):',
      dict(sorted(prefix_dist.items(), key=lambda kv: -kv[1])))
print('rows whose status does not start with a set member:',
      bad_prefix if bad_prefix else 'NONE')
print('STRUCK rows whose prefix says work remains (asymmetric by design):',
      strike_contra if strike_contra else 'NONE')


# ---------------------------------------------------------------------------
# ID UNIQUENESS -- a row id must name exactly one row.
#
# WHY: GAP-227 was allocated twice, 19 minutes apart on 2026-08-20 -- 07386357
# named it deliberately (k_quants clippy, Tier A); 08ed3734, a commit about
# GAP-225, reused it for an unrelated max_ulp row in Tier B. Every individual
# row parses, so every other check in this file passes. Nothing here could see
# it.
#
# AND THE AMBIGUITY REACHED THE CODE BEFORE ANYONE NOTICED: docs/method-rules.md
# cites GAP-227 meaning the clippy gate, while fkc/verify/exact_ref.rs and
# fkc/verify/seed_cpu_ledger.rs both cite it meaning the float8/max_ulp row.
# Two senses, four citations, one id.
#
# *** MATCH THE FULL ID, INCLUDING ANY `(x)` SUFFIX. ***
#
# The obvious implementation -- `grep -oE 'GAP-[0-9]+'` -- TRUNCATES
# `GAP-228(a)` to `GAP-228` and reports EIGHT duplicates where there is one
# parent and seven legitimate lettered sub-rows. Measured on this file: the
# naive form flags {GAP-228: 8, GAP-227: 2}; the full form flags {GAP-227: 2}.
#
# THAT MATTERS MORE THAN THE FALSE POSITIVE ITSELF. A gate that fires on valid
# input teaches its own removal -- the next person to hit it will loosen the
# rule, because the data is obviously fine and the gate is obviously wrong.
# If this check ever flags a lettered sub-row, FIX THE MATCHER, DO NOT WIDEN
# THE RULE.

FULL_ID = re.compile(r'^\| (~*)(GAP-[0-9]+(?:\([a-z0-9]+\))?)')


def duplicate_ids(src):
    """Ids appearing on more than one row, matched in FULL."""
    seen = {}
    for _l in src:
        _m = FULL_ID.match(_l)
        if _m:
            seen.setdefault(_m.group(2), []).append(_l[:60])
    return {k: v for k, v in seen.items() if len(v) > 1}


# FOUNDATION CHECK, both directions. A duplicate must flag AND a lettered
# family must not -- the second arm is the one that would have shipped broken.
_ID_DUP = ['| GAP-900 | a | A | a | **OPEN** |',
           '| GAP-900 | b | B | b | **OPEN** |']
_ID_SUB = ['| GAP-901 | p | A | p | **OPEN** |',
           '| GAP-901(a) | x | A | x | **OPEN** |',
           '| GAP-901(b) | y | A | y | **OPEN** |']
id_foundation = []
if list(duplicate_ids(_ID_DUP)) != ['GAP-900']:
    id_foundation.append('a real duplicate was NOT flagged')
if duplicate_ids(_ID_SUB):
    id_foundation.append('lettered sub-rows WERE flagged (matcher is truncating)')

dup_ids = duplicate_ids(lines)

print()
print('id-uniqueness self-test (duplicate must flag, lettered sub-rows must not):',
      'PASS' if not id_foundation else id_foundation)
print('row ids appearing more than once:',
      {k: len(v) for k, v in dup_ids.items()} if dup_ids else 'NONE')
for _k, _v in sorted(dup_ids.items()):
    for _r in _v:
        print('    %s' % _r)

if (odd or no_pipe or header_problems or control_chars or conflict_markers
        or missing_backlinks or _foundation
        or arity_regression or bad_prefix or strike_contra
        or dup_ids or id_foundation
        or vocab_foundation):
    sys.exit(1)
