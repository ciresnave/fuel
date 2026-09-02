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

lines = io.open('docs/gaps.md', encoding='utf-8').read().split('\n')


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
    return sum(1 for k, ch in enumerate(l)
               if ch == '|' and (k == 0 or l[k - 1] != '\\'))


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

if odd or no_pipe or header_problems:
    sys.exit(1)
