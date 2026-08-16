# Stage 2 bag-rule audit

Status: **not ready to freeze**  
Audit date: 2026-08-16  
New rules added: 0

## Scope and method

The 26 legacy rules whose `recommended_category` is `accessories` were reviewed for:

- an explicit bag recommendation in the rule text;
- direct support from the cited source;
- applicability to the frozen query categories; and
- eligibility for fail-closed bag retrieval.

The row-level decisions and source locators are recorded in
`data/kb/bag_rule_audit.csv`. A rule is retrievable for `bags` only when its audit status is
`retain`, its citation support is `direct`, and its query category is explicitly applicable.

## Result

| Measure | Result |
|---|---:|
| Legacy accessory-target rules audited | 26 |
| Bag-explicit rule texts | 11 |
| Directly supported bag rules | 5 |
| Rules currently approved for retrieval | 0 |
| New-taxonomy query categories with approved coverage | 0 of 4 |
| Freeze ready | No |

Seven of the eleven bag-explicit rules are stored with `input_category=dresses`, which is
outside the frozen taxonomy. The four remaining candidates consist of three outerwear rules
and one shoe rule; all require replacement evidence or a narrower rewrite before approval.

## Unsupported contexts requiring a decision

Because bag-to-bag cases are excluded, bag recommendations can be queried from four categories.

| Query category | Approved applicable rules | Current disposition |
|---|---:|---|
| tops | 0 | Unsupported |
| bottoms | 0 | Unsupported |
| shoes | 0 | `R046` requires rewriting or replacement evidence |
| outerwear | 0 | `R042` and `R043` lack direct support; `R044` requires narrowing |

There is no approved coverage for menswear, womenswear, general/unisex, casual, smart-casual,
business, formal or evening contexts under the new query taxonomy. Those dimensions therefore
remain unsupported rather than being inferred from generic accessory evidence.

## Deferred empirical audit

No prepared five-category dataset or deterministic 200-case bag sample exists in the current
runtime. Consequently rule frequency, concentration, packet duplication and pairwise overlap
cannot yet be measured. The retrieval implementation now fails closed, so all 200 bag cases
would be reported as unsupported until rules are approved. After a revised KB is approved,
Stage 1 data preparation must be run and the case-level audit must confirm coverage before the
KB can be frozen.

## Required approval

Choose one of the following before the KB changes:

1. source and add/rewrite bag rules to cover the four query categories and declared contexts; or
2. explicitly exclude unsupported query/context combinations and report the resulting scope.

No final-condition results should be inspected before this decision and the subsequent KB freeze.
