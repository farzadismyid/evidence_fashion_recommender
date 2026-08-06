# Rule retrieval diversity audit

Packets analysed: 900 (one frozen packet per case/generator; four variant copies were deduplicated). 74/126 KB rules were retrieved. Shannon entropy: 3.795 nats. Mean unique rules/case: 5.00.

## Top ten rules

| rule_id | retrievals | percent_packets | rule_text |
|---|---:|---:|---|
| R025 | 180 | 20.00 | For an outfit retrieved as a complete look, recommend shoes from a different category but compatible with the existing items because complete-the-look systems model complementary rather than substitute items. |
| R050 | 180 | 20.00 | For fashion-item recommendation, treat accessories as complementary items rather than substitutes when completing an outfit around an existing garment. |
| R093 | 180 | 20.00 | For formal shoes as the query item, recommend formal trousers or tailored bottoms because footwear formality should match the rest of the outfit. |
| R075 | 180 | 20.00 | For occasion-aware recommendation, choose tops whose use case matches the event because fashion knowledge links clothing choices to social activity and convention. |
| R074 | 180 | 20.00 | For a top recommendation in a complete outfit, enforce type diversity: recommend tops as complementary items only when the existing item is from another category. |
| R068 | 180 | 20.00 | For a structured bag or polished accessory, recommend a top with a similarly polished finish, such as a blouse, shirt or fine knit. |
| R096 | 180 | 20.00 | For an outerwear-led casual outfit, recommend bottoms that match the jacket's relaxed formality, such as jeans or casual trousers. |
| R101 | 180 | 20.00 | For colour-compatible outfit completion, recommend bottoms whose dominant colour is compatible with the existing top and accessories. |
| R122 | 180 | 20.00 | For refined accessories such as a structured bag or evening jewellery, recommend outerwear with equivalent polish, such as a tailored coat or blazer. |
| R125 | 180 | 20.00 | For outerwear recommendation in a complete-look system, select outerwear as a complementary category item rather than treating it as a substitute for tops or dresses. |

Top-1/top-5/top-10 retrieval shares are in `concentration.csv`; per-rule/category frequencies, overlap, and packet diversity are machine-readable beside this report. High-frequency rules are flagged descriptively by frequency, not declared invalid: category matching is an intentional retrieval restriction and repeated broad compatibility rules may be legitimate. Candidate-specific variation cannot be isolated fully because each frozen packet is for one locked candidate; compare packet diversity/within-category overlap rather than interpreting repetition as proof of incompatibility.
