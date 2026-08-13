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
rows = []
for l in lines:
    if re.match(r'^\| ~*GAP-', l):
        # count pipes that are NOT escaped (not preceded by a backslash)
        delim = 0
        for k, ch in enumerate(l):
            if ch == '|' and (k == 0 or l[k - 1] != '\\'):
                delim += 1
        rid = re.match(r'^\| (~*GAP-[0-9]+)', l).group(1)
        rows.append((rid, delim, l))

struck = [r for r in rows if r[0].startswith('~~')]
live = [r for r in rows if not r[0].startswith('~~')]
closed_by_status = [r for r in live if 'CLOSED' in r[2]]

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
print('rows, live but whose STATUS CELL says CLOSED:                 %d' % len(closed_by_status))
print()
# NOTE: ASCII ONLY below. The first version of this block used an emoji and
# died on cp1252 stdout -- making the gate exit 1 for a PRINTING failure with no
# table violation present. A false red from a gate is worse than no gate: it is
# the failure this project keeps recording, pointed at the instrument itself.
print('!! TWO CLOSE CONVENTIONS COEXIST (strikethrough vs the word CLOSED in the')
print('   status cell), so "how many are open" has no single answer from this')
print('   table alone. Quote the construct with the number.')

if odd or no_pipe:
    sys.exit(1)
